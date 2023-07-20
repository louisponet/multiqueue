
use std::any::Any;
use std::cell::Cell;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, fence};
use std::sync::atomic::Ordering::*;
use std::sync::mpsc::{TrySendError, TryRecvError, RecvError};

use alloc;
use atomicsignal::LoadedSignal;
use countedindex::{CountedIndex, get_valid_wrap, is_tagged, rm_tag, Index, INITIAL_QUEUE_FLAG};
use memory::{MemoryManager, MemToken};
use wait::*;

use read_cursor::{ReadCursor, Reader};

extern crate atomic_utilities;
extern crate smallvec;

use self::atomic_utilities::artificial_dep::{dependently_mut, DepOrd};

/// This is basically acting as a static bool
/// so the queue can act as a normal mpmc in other circumstances
pub trait QueueRW<T> {
    fn inc_ref(_: &AtomicUsize);
    fn dec_ref(_: &AtomicUsize);
    fn check_ref(_: &AtomicUsize) -> bool;
    fn do_drop() -> bool;
    unsafe fn get_val(_: &mut T) -> T;
    fn forget_val(_: T);
    unsafe fn drop_in_place(_: &mut T);
}

#[derive(Clone)]
pub struct BCast<T> {
    mk: PhantomData<T>,
}

impl<T: Clone> QueueRW<T> for BCast<T> {
    // TODO: Skip refcount when type is copyable or clone is safe on junk data
    #[inline(always)]
    fn inc_ref(r: &AtomicUsize) {
        r.fetch_add(1, atomic_utilities::fence_rmw::RMWOrder);
        atomic_utilities::fence_rmw::fence_rmw();
    }

    // TODO: Skip refcount when type is copyable or clone is safe on junk data
    #[inline(always)]
    fn dec_ref(r: &AtomicUsize) {
        r.fetch_sub(1, Relaxed);
    }

    #[inline(always)]
    fn check_ref(r: &AtomicUsize) -> bool {
        r.load(Relaxed) == 0
    }

    #[inline(always)]
    fn do_drop() -> bool {
        true
    }

    #[inline(always)]
    unsafe fn get_val(val: &mut T) -> T {
        val.clone()
    }

    #[inline(always)]
    fn forget_val(_v: T) {}

    #[inline(always)]
    unsafe fn drop_in_place(_v: &mut T) {}
}

#[derive(Clone)]
pub struct MPMC<T> {
    mk: PhantomData<T>,
}

impl<T> QueueRW<T> for MPMC<T> {
    #[inline(always)]
    fn inc_ref(_r: &AtomicUsize) {}

    #[inline(always)]
    fn dec_ref(_r: &AtomicUsize) {}

    #[inline(always)]
    fn check_ref(_r: &AtomicUsize) -> bool {
        true
    }

    #[inline(always)]
    fn do_drop() -> bool {
        false
    }

    #[inline(always)]
    unsafe fn get_val(val: &mut T) -> T {
        ptr::read(val)
    }

    #[inline(always)]
    fn forget_val(val: T) {
        mem::forget(val);
    }

    #[inline(always)]
    unsafe fn drop_in_place(val: &mut T) {
        ptr::drop_in_place(val);
    }
}

#[derive(Clone, Copy)]
pub enum QueueState {
    Uni,
    Multi,
}

/// This holds entries in the queue
struct QueueEntry<T> {
    val: T,
    wraps: AtomicUsize,
}

/// This holds the refcount object
struct RefCnt {
    refcnt: AtomicUsize,
    _buffer: [u8; 64],
}

/// A bounded queue that supports multiple reader and writers
/// and supports effecient methods for single consumers and producers
#[repr(C)]
pub struct MultiQueue<RW: QueueRW<T>, T> {
    d1: [u8; 64],

    // Writer data
    head: CountedIndex,
    tail_cache: AtomicUsize,
    writers: AtomicUsize,
    d2: [u8; 64],

    // Shared Data
    // The data and the wraps flag are in the same location
    // to reduce the # of distinct cache lines read when getting an item
    // The tail itself is rarely modified, making it a suitable candidate
    // to be in the shared space
    tail: ReadCursor,
    data: *mut QueueEntry<T>,
    refs: *mut RefCnt,
    capacity: isize,
    pub waiter: Arc<dyn Wait>,
    needs_notify: bool,
    mk: PhantomData<RW>,
    d3: [u8; 64],

    pub manager: MemoryManager,
    d4: [u8; 64],
}

pub struct InnerSend<RW: QueueRW<T>, T> {
    queue: Arc<MultiQueue<RW, T>>,
    token: *const MemToken,
    state: Cell<QueueState>,
}

pub struct InnerRecv<RW: QueueRW<T>, T> {
    queue: Arc<MultiQueue<RW, T>>,
    reader: Reader,
    token: *const MemToken,
    alive: bool,
}

/// This is a sender that can transparently act as a futures stream

impl<RW: QueueRW<T>, T> MultiQueue<RW, T> {
    pub fn new(_capacity: Index) -> (InnerSend<RW, T>, InnerRecv<RW, T>) {
        MultiQueue::new_with(_capacity, BlockingWait::new())
    }

    pub fn new_with<W: Wait + 'static>(capacity: Index,
                                       wait: W)
                                       -> (InnerSend<RW, T>, InnerRecv<RW, T>) {
        MultiQueue::new_internal(capacity, Arc::new(wait))
    }

    fn new_internal(_capacity: Index, wait: Arc<dyn Wait>) -> (InnerSend<RW, T>, InnerRecv<RW, T>) {
        let capacity = get_valid_wrap(_capacity);
        let queuedat: *mut QueueEntry<T> = alloc::allocate(capacity as usize);
        let refdat: *mut RefCnt = alloc::allocate(capacity as usize);
        unsafe {
            for i in 0..capacity as isize {
                let elem: &QueueEntry<T> = &*queuedat.offset(i);
                elem.wraps.store(INITIAL_QUEUE_FLAG, Relaxed);

                let refd: &RefCnt = &*refdat.offset(i);
                refd.refcnt.store(0, Relaxed);
            }
        }

        let (cursor, reader) = ReadCursor::new(capacity);
        let needs_notify = wait.needs_notify();
        let queue = MultiQueue {
            d1: [0;64],

            head: CountedIndex::new(capacity),
            tail_cache: AtomicUsize::new(0),
            writers: AtomicUsize::new(1),
            d2: [0;64],

            tail: cursor,
            data: queuedat,
            refs: refdat,
            capacity: capacity as isize,
            waiter: wait,
            needs_notify: needs_notify,
            mk: PhantomData,
            d3: [0;64],

            manager: MemoryManager::new(),

            d4: [0;64],
        };

        let qarc = Arc::new(queue);

        let mwriter = InnerSend {
            queue: qarc.clone(),
            state: Cell::new(QueueState::Uni),
            token: qarc.manager.get_token(),
        };

        let mreader = InnerRecv {
            queue: qarc.clone(),
            reader: reader,
            token: qarc.manager.get_token(),
            alive: true,
        };

        (mwriter, mreader)
    }

    pub fn try_send_multi(&self, val: T) -> Result<(), TrySendError<T>> {
        let mut transaction = self.head.load_transaction(Relaxed);

        unsafe {
            loop {
                let (chead, wrap_valid_tag) = transaction.get();
                let tail_cache = self.tail_cache.load(Relaxed);
                if transaction.matches_previous(tail_cache) {
                    let new_tail = self.reload_tail_multi(tail_cache, wrap_valid_tag);
                    if transaction.matches_previous(new_tail) {
                        return Err(TrySendError::Full(val));
                    }
                }
                let write_cell = &mut *self.data.offset(chead);
                let ref_cell = &*self.refs.offset(chead);
                if !RW::check_ref(&ref_cell.refcnt) {
                    return Err(TrySendError::Full(val));
                }
                fence(Acquire);

                match transaction.commit(1, Relaxed) {
                    Some(new_transaction) => transaction = new_transaction,
                    None => {
                        let current_tag = write_cell.wraps.load(Relaxed);

                        // This will delay the dropping of the exsisting item until
                        // after the write is done. This will have a marginal effect on
                        // throughput in most cases but will really help latency.
                        // Hopefully the compiler is smart enough to get rid of this
                        // when there's no drop
                        let _possible_drop = if RW::do_drop() && !is_tagged(current_tag) {
                            Some(ptr::read(&write_cell.val))
                        } else {
                            None
                        };
                        ptr::write(&mut write_cell.val, val);
                        write_cell.wraps.store(wrap_valid_tag, Release);
                        return Ok(());
                    }
                }
            }
        }
    }

    pub fn try_send_single(&self, val: T) -> Result<(), TrySendError<T>> {
        let transaction = self.head.load_transaction(Relaxed);
        let (chead, wrap_valid_tag) = transaction.get();
        unsafe {
            let tail_cache = self.tail_cache.load(Relaxed);
            if transaction.matches_previous(tail_cache) {
                let new_tail = self.reload_tail_single(wrap_valid_tag);
                if transaction.matches_previous(new_tail) {
                    return Err(TrySendError::Full(val));
                }
            }
            let write_cell = &mut *self.data.offset(chead);
            let ref_cell = &*self.refs.offset(chead);
            if !RW::check_ref(&ref_cell.refcnt) {
                return Err(TrySendError::Full(val));
            }
            fence(Acquire);
            transaction.commit_direct(1, Relaxed);
            let current_tag = write_cell.wraps.load(Relaxed);
            let _possible_drop = if RW::do_drop() && !is_tagged(current_tag) {
                Some(ptr::read(&write_cell.val))
            } else {
                None
            };
            ptr::write(&mut write_cell.val, val);
            write_cell.wraps.store(wrap_valid_tag, Release);
            Ok(())
        }
    }

    pub fn try_recv(&self, reader: &Reader) -> Result<T, (*const AtomicUsize, TryRecvError)> {
        let mut ctail_attempt = reader.load_attempt(Relaxed);
        let is_single = reader.is_single();
        unsafe {
            loop {
                let (ctail, wrap_valid_tag) = ctail_attempt.get();
                let read_cell = &mut *self.data.offset(ctail);

                // For any curious readers, this gnarly if block catchs a race between
                // advancing the write index and unsubscribing from the queue. in short,
                // Since unsubscribe happens after the read_cell is written, there's a race
                // between the first and second if statements. Hence, a second check is required
                // after the writer load so ensure that the the wrap_valid_tag is still wrong so
                // we had actually seen a race. Doing it this way removes fences on the fast path
                let seen_tag = read_cell.wraps.load(DepOrd);
                if rm_tag(seen_tag) != wrap_valid_tag {
                    if self.writers.load(Relaxed) == 0 {
                        fence(Acquire);
                        if rm_tag(read_cell.wraps.load(Acquire)) != wrap_valid_tag {
                            return Err((ptr::null(), TryRecvError::Disconnected));
                        }
                    }
                    return Err((&read_cell.wraps, TryRecvError::Empty));
                }
                let ref_cell = &*self.refs.offset(ctail);
                if !is_single {
                    RW::inc_ref(&ref_cell.refcnt);
                    if reader.load_count(Relaxed) != wrap_valid_tag {
                        RW::dec_ref(&ref_cell.refcnt);
                        ctail_attempt = ctail_attempt.reload();
                        continue;
                    }
                }
                let rval = dependently_mut(seen_tag, &mut read_cell.val, |rc| RW::get_val(rc));
                fence(Release);
                if !is_single {
                    RW::dec_ref(&ref_cell.refcnt);
                }
                match ctail_attempt.commit_attempt(1, Relaxed) {
                    Some(new_attempt) => {
                        ctail_attempt = new_attempt;
                        RW::forget_val(rval);
                    }
                    None => return Ok(rval),
                }
            }
        }
    }

    pub fn try_recv_view<R, F: FnOnce(&T) -> R>
        (&self,
         op: F,
         reader: &Reader)
         -> Result<R, (F, *const AtomicUsize, TryRecvError)> {
        let ctail_attempt = reader.load_attempt(Relaxed);
        unsafe {
            let (ctail, wrap_valid_tag) = ctail_attempt.get();
            let read_cell = &mut *self.data.offset(ctail);
            let seen_tag = rm_tag(read_cell.wraps.load(DepOrd));
            if seen_tag != wrap_valid_tag {
                if self.writers.load(Relaxed) == 0 {
                    fence(Acquire);
                    if rm_tag(read_cell.wraps.load(Acquire)) != wrap_valid_tag {
                        return Err((op, ptr::null(), TryRecvError::Disconnected));
                    }
                }
                return Err((op, &read_cell.wraps, TryRecvError::Empty));
            }
            dependently_mut(seen_tag, &mut read_cell.val, |rv_ref| {
                let rval = op(rv_ref);
                RW::drop_in_place(rv_ref);
                ctail_attempt.commit_direct(1, Release);
                Ok(rval)
            })
        }
    }

    fn reload_tail_multi(&self, tail_cache: usize, count: usize) -> usize {
        if let Some(max_diff_from_head) = self.tail.get_max_diff(count) {
            let current_tail = CountedIndex::get_previous(count, max_diff_from_head);
            if tail_cache == current_tail {
                return current_tail;
            }
            match self.tail_cache.compare_exchange(tail_cache, current_tail, AcqRel, Relaxed) {
                Ok(_) => current_tail,
                Err(val) => val,
            }
        } else {
            self.tail_cache.load(Acquire)
        }
    }

    fn reload_tail_single(&self, count: usize) -> usize {
        let max_diff_from_head = self.tail
            .get_max_diff(count)
            .expect("The write head got ran over by consumers in single writer mode. This \
                     process is borked!");
        let current_tail = CountedIndex::get_previous(count, max_diff_from_head);
        self.tail_cache.store(current_tail, Relaxed);
        current_tail
    }
}

impl<RW: QueueRW<T>, T> InnerSend<RW, T> {
    #[inline(always)]
    pub fn try_send(&self, val: T) -> Result<(), TrySendError<T>> {
        let signal = self.queue
            .manager
            .signal
            .load(Relaxed);
        if signal.has_action() {
            let disconnected = self.handle_signals(signal);
            if disconnected {
                return Err(TrySendError::Full(val));
            }
        }
        let val = match self.state.get() {
            QueueState::Uni => self.queue.try_send_single(val),
            QueueState::Multi => {
                if self.queue.writers.load(Relaxed) == 1 {
                    fence(Acquire);
                    self.state.set(QueueState::Uni);
                    self.queue.try_send_single(val)
                } else {
                    self.queue.try_send_multi(val)
                }
            }
        };
        // Putting this in the send functions
        // greatly confuses the compiler and literally halfs
        // the performance of the queue. I suspect the compiler
        // always sets up a stack from regardless of the condition
        // and that hurts optimizations around it.
        if val.is_ok() {
            if self.queue.needs_notify {
                self.queue.waiter.notify();
            }
        }
        val
    }

    /// Removes the writer as a producer to the queue
    pub fn unsubscribe(self) {}

    #[cold]
    fn handle_signals(&self, signal: LoadedSignal) -> bool {
        if signal.get_epoch() {
            self.queue.manager.update_token(self.token);
        }
        signal.get_reader()
    }
}

impl<RW: QueueRW<T>, T> InnerRecv<RW, T> {
    #[inline(always)]
    pub fn try_recv(&self) -> Result<T, TryRecvError> {
        self.examine_signals();
        match self.queue.try_recv(&self.reader) {
            Ok(v) => Ok(v),
            Err((_, e)) => Err(e),
        }
    }

    pub fn recv(&self) -> Result<T, RecvError> {
        self.examine_signals();
        loop {
            match self.queue.try_recv(&self.reader) {
                Ok(v) => return Ok(v),
                Err((_, TryRecvError::Disconnected)) => return Err(RecvError),
                Err((pt, TryRecvError::Empty)) => {
                    let count = self.reader.load_count(Relaxed);
                    unsafe {
                        self.queue.waiter.wait(count, &*pt, &self.queue.writers);
                    }
                }
            }
        }
    }

    pub fn is_single(&self) -> bool {
        self.reader.get_consumers() == 1
    }

    #[inline(always)]
    pub fn try_recv_view<R, F: FnOnce(&T) -> R>(&self, op: F) -> Result<R, (F, TryRecvError)> {
        self.examine_signals();
        match self.queue.try_recv_view(op, &self.reader) {
            Ok(v) => Ok(v),
            Err((op, _, e)) => Err((op, e)),
        }
    }

    pub fn recv_view<R, F: FnOnce(&T) -> R>(&self, mut op: F) -> Result<R, (F, RecvError)> {
        self.examine_signals();
        loop {
            match self.queue.try_recv_view(op, &self.reader) {
                Ok(v) => return Ok(v),
                Err((o, _, TryRecvError::Disconnected)) => return Err((o, RecvError)),
                Err((o, pt, TryRecvError::Empty)) => {
                    op = o;
                    let count = self.reader.load_count(Relaxed);
                    unsafe {
                        self.queue.waiter.wait(count, &*pt, &self.queue.writers);
                    }
                }
            }
        }
    }

    pub fn add_stream(&self) -> InnerRecv<RW, T> {
        InnerRecv {
            queue: self.queue.clone(),
            reader: self.queue.tail.add_stream(&self.reader, &self.queue.manager),
            token: self.queue.manager.get_token(),
            alive: true,
        }
    }

    #[inline(always)]
    fn examine_signals(&self) {
        let signal = self.queue
            .manager
            .signal
            .load(Relaxed);
        if signal.has_action() {
            self.handle_signals(signal);
        }
    }

    #[cold]
    fn handle_signals(&self, signal: LoadedSignal) {
        if signal.get_epoch() {
            self.queue.manager.update_token(self.token);
        }
    }


    pub fn unsubscribe(self) -> bool {
        self.reader.get_consumers() == 1
    }

    /// Runs the passed function after unsubscribing the reader from the queue
    unsafe fn do_unsubscribe_with<F: FnOnce()>(&mut self, f: F) {
        if self.alive {
            self.alive = false;
            if self.reader.remove_consumer() == 1 {
                if self.queue.tail.remove_reader(&self.reader, &self.queue.manager) {
                    self.queue
                        .manager
                        .signal
                        .set_reader(SeqCst);
                }
                self.queue.manager.remove_token(self.token);
            }
            fence(SeqCst);
            f()
        }
    }
}



// The mpsc SendError struct can't be constructed according to rustc
// since it's a struct and the ctor is private. Copied and pasted here

/// Error type for sending, used when the receiving end of the channel is
/// dropped
pub struct SendError<T>(T);

impl<T> fmt::Debug for SendError<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_tuple("SendError").field(&"...").finish()
    }
}

impl<T> fmt::Display for SendError<T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "send failed because receiver is gone")
    }
}

impl<T> Error for SendError<T>
    where T: Any
{
    fn description(&self) -> &str {
        "send failed because receiver is gone"
    }
}

impl<T> SendError<T> {
    /// Returns the message that was attempted to be sent but failed.
    pub fn into_inner(self) -> T {
        self.0
    }
}

//////// Clone implementations

impl<RW: QueueRW<T>, T> Clone for InnerSend<RW, T> {
    fn clone(&self) -> InnerSend<RW, T> {
        self.state.set(QueueState::Multi);
        let rval = InnerSend {
            queue: self.queue.clone(),
            state: Cell::new(QueueState::Multi),
            token: self.queue.manager.get_token(),
        };
        self.queue.writers.fetch_add(1, SeqCst);
        rval
    }
}

impl<RW: QueueRW<T>, T> Clone for InnerRecv<RW, T> {
    fn clone(&self) -> InnerRecv<RW, T> {
        self.reader.dup_consumer();
        InnerRecv {
            queue: self.queue.clone(),
            reader: self.reader.clone(),
            token: self.queue.manager.get_token(),
            alive: true,
        }
    }
}

//////// Drop implementations

impl<RW: QueueRW<T>, T> Drop for InnerSend<RW, T> {
    fn drop(&mut self) {
        self.queue.writers.fetch_sub(1, SeqCst);
        fence(SeqCst);
        self.queue.manager.remove_token(self.token);
        self.queue.waiter.notify();
    }
}

impl<RW: QueueRW<T>, T> Drop for InnerRecv<RW, T> {
    fn drop(&mut self) {
        unsafe { self.do_unsubscribe_with(|| ()) }
    }
}

impl<RW: QueueRW<T>, T> Drop for MultiQueue<RW, T> {
    fn drop(&mut self) {
        if RW::do_drop() {
            // everything that's tagged shouldn't be dropped
            // otherwise, everything else is valid and waiting to be read
            // or invalid and waiting to be overwritten/dropped
            for i in 0..self.capacity {
                unsafe {
                    let cell = &mut *self.data.offset(i);
                    if !is_tagged(cell.wraps.load(Relaxed)) {
                        ptr::read(&cell.val);
                    }
                }
            }
        } else {
            let last_read = CountedIndex::from_usize(self.tail.last_pos.get(),
                                                     self.capacity as Index);
            while last_read.load_count(Relaxed) != self.head.load_count(Relaxed) {
                unsafe {
                    let cur_pos = last_read.load_transaction(Relaxed);
                    let (cur_ind, _) = cur_pos.get();
                    ptr::drop_in_place(&mut (*self.data.offset(cur_ind)).val);
                    cur_pos.commit_direct(1, Relaxed);
                }
            }
        }
    }
}
impl<RW: QueueRW<T>, T> fmt::Debug for InnerRecv<RW, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "MultiQueue error message - you probably tried to unwrap the result of into_single")
    }
}


unsafe impl<RW: QueueRW<T>, T> Sync for MultiQueue<RW, T> {}
unsafe impl<RW: QueueRW<T>, T> Send for MultiQueue<RW, T> {}
unsafe impl<RW: QueueRW<T>, T> Send for InnerSend<RW, T> {}
unsafe impl<RW: QueueRW<T>, T> Send for InnerRecv<RW, T> {}
