//! Poison-tolerant locking. A panic on one connection thread must not wedge the lane for every
//! later caller: the state guarded here is counters and tables that are consistent between
//! statements, so a poisoned lock is recovered instead of re-raising the panic.

use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// `Mutex::lock` that recovers from poisoning.
pub(crate) trait MutexExt<T> {
    fn plock(&self) -> MutexGuard<'_, T>;
}

impl<T> MutexExt<T> for Mutex<T> {
    fn plock(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// `RwLock::read` / `write` that recover from poisoning.
pub(crate) trait RwLockExt<T> {
    fn pread(&self) -> RwLockReadGuard<'_, T>;
    fn pwrite(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T> RwLockExt<T> for RwLock<T> {
    fn pread(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap_or_else(PoisonError::into_inner)
    }
    fn pwrite(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap_or_else(PoisonError::into_inner)
    }
}
