//! Cross-process file-locking behaviour probe.
//!
//! Measures how this platform actually behaves — advisory vs mandatory
//! locks, shared/exclusive compatibility, and crash release — instead of
//! trusting documentation. The results decide the locking contract in the
//! format spec's Concurrency section.
//!
//! Run on each platform of interest and compare the reports:
//!
//! ```text
//! cargo run --example lock_probe
//! ```
//!
//! The probe spawns itself as a child process for the cross-process cases,
//! so a single command produces the whole report. Uses only std (file
//! locking is stable since Rust 1.89). No .htt files are involved; this is
//! purely about the platform primitive the format will standardize on.
//!
//! Expected on Unix (advisory flock): unlocked reads and writes SUCCEED
//! while another process holds an exclusive lock — the lock binds only
//! processes that ask. The open question this probe answers on Windows
//! (LockFileEx) is whether those same unlocked operations FAIL, i.e.
//! whether the kernel enforces the lock against non-participants.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("child-ops") => child_ops(&args[2]),
        Some("child-crash") => child_crash(&args[2]),
        _ => parent(),
    }
}

/// Child mode: perform every operation WITHOUT coordinating with the parent's
/// lock, and report what the OS allows. One line per operation.
fn child_ops(path: &str) {
    match std::fs::read(path) {
        Ok(bytes) => println!("  child unlocked-read        -> Ok({} bytes)", bytes.len()),
        Err(e) => println!("  child unlocked-read        -> Err({:?})", e.kind()),
    }

    match OpenOptions::new().append(true).open(path) {
        Ok(mut f) => match f.write_all(b"x") {
            Ok(()) => println!("  child unlocked-append      -> Ok"),
            Err(e) => println!("  child unlocked-append      -> Err({:?})", e.kind()),
        },
        Err(e) => println!("  child open-for-append      -> Err({:?})", e.kind()),
    }

    match OpenOptions::new().read(true).write(true).open(path) {
        Ok(f) => {
            println!("  child try_lock (exclusive) -> {:?}", f.try_lock().map(|_| ()));
            println!("  child try_lock_shared      -> {:?}", f.try_lock_shared().map(|_| ()));
        }
        Err(e) => println!("  child open-for-lock        -> Err({:?})", e.kind()),
    }
}

/// Child mode: take an exclusive lock and exit WITHOUT unlocking, simulating
/// a crashed writer. The parent then checks whether the OS released the lock.
fn child_crash(path: &str) {
    let f = OpenOptions::new().read(true).write(true).open(path).unwrap();
    match f.try_lock() {
        Ok(_) => {
            println!("  crash-child took exclusive -> Ok (now exiting without unlock)");
            std::process::exit(0);
        }
        Err(e) => {
            println!("  crash-child took exclusive -> Err({:?})", e);
            std::process::exit(1);
        }
    }
}

fn run_child(mode: &str, path: &str) {
    let exe = std::env::current_exe().expect("current_exe");
    let out = Command::new(exe)
        .arg(mode)
        .arg(path)
        .output()
        .expect("spawn child");
    print!("{}", String::from_utf8_lossy(&out.stdout));
    let err = String::from_utf8_lossy(&out.stderr);
    if !err.trim().is_empty() {
        print!("  child stderr: {}", err);
    }
}

fn parent() {
    println!("=== lock_probe report ===");
    println!("os: {} / arch: {} / family: {}", std::env::consts::OS, std::env::consts::ARCH, std::env::consts::FAMILY);

    let dir = std::env::temp_dir();
    let path = dir.join(format!("htt_lock_probe_{}.bin", std::process::id()));
    let path_str = path.to_string_lossy().to_string();
    std::fs::write(&path, b"lock-probe-payload").unwrap();

    // Phase A: parent holds EXCLUSIVE, child tries everything uncoordinated.
    {
        let holder = File::options().read(true).write(true).open(&path).unwrap();
        holder
            .try_lock()
            .expect("parent could not take the exclusive lock on a fresh file");
        println!("\n[A] parent holds EXCLUSIVE:");
        run_child("child-ops", &path_str);
        holder.unlock().expect("unlock after phase A");
    }

    // Phase B: parent holds SHARED.
    {
        let holder = File::options().read(true).write(true).open(&path).unwrap();
        holder
            .try_lock_shared()
            .expect("parent could not take the shared lock");
        println!("\n[B] parent holds SHARED:");
        run_child("child-ops", &path_str);
        holder.unlock().expect("unlock after phase B");
    }

    // Phase C: crash release — child locks exclusively and dies unlocked.
    {
        println!("\n[C] crash release:");
        run_child("child-crash", &path_str);
        let f = File::options().read(true).write(true).open(&path).unwrap();
        println!(
            "  parent try_lock after child death -> {:?} (Ok = kernel auto-released)",
            f.try_lock().map(|_| ())
        );
        let _ = f.unlock();
    }

    // Phase D: single-process compatibility matrix, as a sanity baseline.
    {
        println!("\n[D] same-host two-handle matrix:");
        let a = File::options().read(true).write(true).open(&path).unwrap();
        let b = File::options().read(true).write(true).open(&path).unwrap();
        a.try_lock().unwrap();
        println!("  a=EX, b try_lock          -> {:?}", b.try_lock().map(|_| ()));
        println!("  a=EX, b try_lock_shared   -> {:?}", b.try_lock_shared().map(|_| ()));
        a.unlock().unwrap();
        a.try_lock_shared().unwrap();
        println!("  a=SH, b try_lock_shared   -> {:?}", b.try_lock_shared().map(|_| ()));
        println!("  a=SH, b try_lock          -> {:?}", b.try_lock().map(|_| ()));
    }

    let _ = std::fs::remove_file(&path);
    println!("\n=== end of report — paste this whole output back ===");
}
