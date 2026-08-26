//! Build script for compiling Metal shaders.
//!
//! This script compiles .metal shader files into .metallib libraries
//! that can be embedded into the binary at compile time.
//!
//! Supports dual compilation:
//! - Metal 3 (macOS 14.0+): All shaders in src/kernels/metal/
//! - Metal 4 (macOS 26.0+): MPP/NAX shaders in src/kernels/metal4/
//!   Only compiled when Metal compiler version >= 400 and SDK >= 26.0

use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Mount point under which macOS 26 exposes the Metal toolchain cryptex.
const CRYPTEX_ROOT: &str = "/private/var/run/com.apple.security.cryptexd/mnt";

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("src").join("kernels").join("metal");
    let metal4_dir = manifest_dir.join("src").join("kernels").join("metal4");

    // Check if we're on macOS
    if env::var("CARGO_CFG_TARGET_OS").unwrap() != "macos" {
        println!("cargo:warning=Metal shaders only compile on macOS");
        return;
    }

    // Resolve the Metal toolchain once and thread it through: metal_toolchain_bin()
    // touches the filesystem, so calling it per shader would be wasted I/O.
    let toolchain = metal_toolchain_bin();
    let toolchain = toolchain.as_deref();

    // Verify Metal toolchain is available before attempting compilation
    check_metal_toolchain(toolchain);

    // Detect Metal compiler version and SDK version
    let metal_version = detect_metal_version(toolchain);
    let sdk_version = detect_sdk_version();

    // Compile Metal 3 shaders (always)
    if shaders_dir.exists() {
        compile_metal_shaders(
            &shaders_dir,
            &out_dir,
            "air64-apple-macos14.0",
            "pmetal_kernels",
            toolchain,
        );
    } else {
        println!(
            "cargo:warning=No metal shaders directory found at {:?}",
            shaders_dir
        );
    }

    // Compile Metal 4 / MPP shaders (conditional on toolchain support)
    let has_metal4 = metal_version >= 400 && sdk_version >= 26.0;
    if has_metal4 && metal4_dir.exists() {
        let metal4_files: Vec<_> = std::fs::read_dir(&metal4_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|e| {
                let p = e.ok()?.path();
                if p.extension()?.to_str()? == "metal" {
                    Some(p)
                } else {
                    None
                }
            })
            .collect();

        if !metal4_files.is_empty() {
            compile_metal_shaders(
                &metal4_dir,
                &out_dir,
                "air64-apple-macos26.0",
                "pmetal_kernels_metal4",
                toolchain,
            );
            println!("cargo:rustc-cfg=has_metal4");
            println!(
                "cargo:warning=Metal 4 / MPP shaders compiled (Metal version {metal_version}, SDK {sdk_version})"
            );
        }
    } else if metal4_dir.exists() {
        // Metal 4 shaders present but toolchain lacks support — expected on SDK < 27.
        // Not a warning; just skip silently.
    }

    // Declare has_metal4 cfg for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_metal4)");

    // Re-run if shaders change
    println!("cargo:rerun-if-changed=src/kernels/metal");
    println!("cargo:rerun-if-changed=src/kernels/metal4");

    // Re-run if the toolchain moves.  The cryptex mount root's mtime changes when
    // a toolchain is installed or removed (e.g. `xcodebuild -downloadComponent
    // MetalToolchain`), so without this cargo would happily reuse a cached build
    // script result that resolved a since-departed toolchain path.
    println!("cargo:rerun-if-env-changed=PMETAL_METAL_TOOLCHAIN_BIN");
    println!("cargo:rerun-if-changed={CRYPTEX_ROOT}");

    // Link against Accelerate.framework for vDSP vector operations
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Conditionally link IOSurface.framework when ANE feature is active
    if env::var("CARGO_FEATURE_ANE").is_ok() {
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        // AppleNeuralEngine.framework is loaded at runtime via dlopen, not linked here
    }
}

/// Detect the maximum Metal language version the toolchain supports.
///
/// `__METAL_VERSION__` is determined by the chosen `-std=` flag. Without one,
/// the compiler falls back to its default (e.g. `metal3.1` on many SDKs even
/// when `metal4.0` is available), so a bare `metal -E` returns 310 on SDK 26+.
/// We probe by requesting `-std=metal4.0` explicitly: if the compiler accepts,
/// it preprocesses `__METAL_VERSION__` to `400`; otherwise it errors and we
/// fall back to probing `metal3.1`. Returns 0 when nothing works (should be
/// rare — means the toolchain is broken, not just old).
fn detect_metal_version(toolchain: Option<&Path>) -> u32 {
    for std_flag in &["metal4.0", "metal3.1"] {
        // Spawned directly rather than through a shell.  An earlier revision built a
        // `zsh -c` pipeline with the toolchain path interpolated into it; single-quoting
        // is not sufficient escaping, since a path containing a single quote closes the
        // quoted token and the remainder is parsed as shell code.  Passing argv straight
        // to the process removes the question entirely.
        let mut cmd = metal_tool_base(toolchain, "metal");
        cmd.arg(format!("-std={std_flag}"))
            .args(["-E", "-x", "metal", "-P", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        let Ok(mut child) = cmd.spawn() else {
            continue;
        };

        // Scoped so the pipe is closed (EOF) before we wait, otherwise the
        // compiler would block reading stdin and we would block reading stdout.
        {
            let Some(mut stdin) = child.stdin.take() else {
                continue;
            };
            if stdin.write_all(b"__METAL_VERSION__\n").is_err() {
                continue;
            }
        }

        let Ok(out) = child.wait_with_output() else {
            continue;
        };
        if !out.status.success() {
            continue;
        }

        // `-P` suppresses line markers, so the expansion is the last non-blank line.
        let text = String::from_utf8_lossy(&out.stdout);
        if let Some(last) = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .next_back()
            && let Ok(n) = last.parse::<u32>()
            && n > 0
        {
            return n;
        }
    }
    0
}

/// Detect macOS SDK version (e.g. 14.5, 26.2).
fn detect_sdk_version() -> f64 {
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "--show-sdk-version"])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let version_str = String::from_utf8_lossy(&o.stdout).trim().to_string();
            version_str.parse::<f64>().unwrap_or(0.0)
        }
        _ => 0.0,
    }
}

/// Verify that the Metal compiler toolchain is available via xcrun.
///
/// Provides actionable installation instructions on failure instead of
/// a cryptic "Failed to run Metal compiler" panic.
fn check_metal_toolchain(toolchain: Option<&Path>) {
    // A directly-resolved toolchain bypasses xcrun entirely, so xcrun's own
    // (per-user) discovery failing is not a build blocker in that case.
    if toolchain.is_some() {
        return;
    }

    // Check xcrun itself
    let xcrun_ok = Command::new("xcrun").args(["--find", "metal"]).output();

    match xcrun_ok {
        Ok(output) if output.status.success() => {} // Metal compiler found
        Ok(_) => {
            // xcrun exists but can't find metal
            panic!(
                "\n\
                 ╔══════════════════════════════════════════════════════════════════╗\n\
                 ║  Metal compiler not found                                       ║\n\
                 ╚══════════════════════════════════════════════════════════════════╝\n\n\
                 PMetal requires the Metal shader compiler to build.\n\n\
                 To install:\n\
                 1. Install Xcode from the App Store (or Command Line Tools):\n\
                    xcode-select --install\n\n\
                 2. Accept the Xcode license:\n\
                    sudo xcodebuild -license accept\n\n\
                 3. Download the Metal toolchain:\n\
                    xcodebuild -downloadComponent MetalToolchain\n\n\
                 4. Restart your terminal (or reboot) after installation.\n\n\
                 If you have Xcode installed but metal is not found, try:\n\
                    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer\n"
            );
        }
        Err(e) => {
            // xcrun itself not found
            panic!(
                "\n\
                 ╔══════════════════════════════════════════════════════════════════╗\n\
                 ║  xcrun not found — Xcode Command Line Tools required            ║\n\
                 ╚══════════════════════════════════════════════════════════════════╝\n\n\
                 PMetal requires Xcode Command Line Tools to compile Metal shaders.\n\n\
                 To install:\n\
                    xcode-select --install\n\n\
                 After installation, restart your terminal and try again.\n\n\
                 Error: {e}\n"
            );
        }
    }
}

/// Directory holding the Metal toolchain binaries (`metal`, `metallib`), if one
/// can be resolved without relying on `xcrun`'s per-user discovery.
///
/// On macOS 26 (Tahoe) Apple moved the Metal compiler out of `Xcode.app` into a
/// separately-downloaded cryptex mounted under
/// `/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/`.
/// `xcrun` locates it through a **per-user** index plist at
/// `~/Library/Developer/Xcode/XcodeToMetalToolchainIndexMapping.plist`. A build
/// running as a service account that has never had that plist written for it —
/// a CI runner, a distribution packager's unprivileged build user — fails with
/// "cannot execute tool 'metal' due to missing Metal Toolchain" even though the
/// toolchain is installed system-wide and works for the login user.
///
/// Resolution order:
/// 1. `PMETAL_METAL_TOOLCHAIN_BIN` — explicit override for packagers.
/// 2. The cryptex mount, when exactly one Metal toolchain is present.
/// 3. `None` — callers fall back to plain `xcrun`, which is correct on
///    macOS < 26 and for ordinary interactive builds.
fn metal_toolchain_bin() -> Option<PathBuf> {
    if let Ok(dir) = env::var("PMETAL_METAL_TOOLCHAIN_BIN") {
        let dir = PathBuf::from(dir);
        if dir.join("metal").is_file() {
            return Some(dir);
        }
        // An override that does not resolve is a packaging error, not something
        // to paper over by silently falling back to a broken `xcrun`.
        panic!(
            "PMETAL_METAL_TOOLCHAIN_BIN is set to {} but no `metal` binary was found there",
            dir.display()
        );
    }

    // Sole cryptex mount, if unambiguous. More than one means we cannot tell
    // which matches the active Xcode, so defer to xcrun rather than guess.
    let mut mounts: Vec<PathBuf> = glob_metal_cryptex_mounts();
    if mounts.len() == 1 {
        return mounts.pop();
    }
    None
}

/// Enumerate `Metal.xctoolchain/usr/bin` directories under the cryptex mount root.
///
/// Hand-rolled rather than pulling in a glob crate: build-dependencies are a
/// supply-chain surface, and this is a single fixed-depth directory scan.
fn glob_metal_cryptex_mounts() -> Vec<PathBuf> {
    const PREFIX: &str = "com.apple.MobileAsset.MetalToolchain-";

    let Ok(entries) = std::fs::read_dir(CRYPTEX_ROOT) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with(PREFIX))
        .map(|e| e.path().join("Metal.xctoolchain/usr/bin"))
        .filter(|p| p.join("metal").is_file())
        .collect();
    // Sorted so the list is deterministic across readdir orderings.  The caller
    // currently only accepts a single mount, but returning an arbitrary order
    // would make any future "pick one" rule silently machine-dependent.
    found.sort();
    found
}

/// Build a bare `Command` for a Metal toolchain tool (`metal`, `metallib`).
///
/// Invokes the tool directly when one was resolved, otherwise goes through
/// `xcrun` exactly as before.
///
/// The direct path deliberately omits xcrun's `-sdk macosx`: that argument tells
/// *xcrun* which SDK to search for the tool, and has no meaning once the tool is
/// being executed by absolute path. Which SDK the compiler builds against is
/// selected by the `-target` triple passed at each call site, not by xcrun.
fn metal_tool_base(toolchain: Option<&Path>, tool: &str) -> Command {
    match toolchain {
        Some(dir) => Command::new(dir.join(tool)),
        None => {
            let mut c = Command::new("xcrun");
            c.args(["-sdk", "macosx", tool]);
            c
        }
    }
}

/// As [`metal_tool_base`], plus the HOME/cache redirection used for shader builds.
fn metal_tool_command(
    toolchain: Option<&Path>,
    tool: &str,
    out_dir: &Path,
    cache_root: &Path,
) -> Command {
    let mut cmd = metal_tool_base(toolchain, tool);
    cmd.env("HOME", xcrun_home(out_dir));
    cmd.env("XDG_CACHE_HOME", cache_root);
    cmd
}

/// Return the HOME directory to use when invoking the Metal compiler or xcrun.
///
/// On macOS 26 (Tahoe), Apple moved the Metal compiler into a cryptex that
/// xcrun discovers via a per-user mapping plist at
/// ~/Library/Developer/Xcode/XcodeToMetalToolchainIndexMapping.plist.
/// If the real HOME has that plist, use it so xcrun can find the compiler.
/// Otherwise fall back to out_dir (isolates xcrun's cache writes from the
/// user's real home directory, which is the right behaviour on macOS < 26).
fn xcrun_home(out_dir: &Path) -> String {
    let real = std::env::var("HOME").unwrap_or_default();
    let mapping = std::path::Path::new(&real)
        .join("Library/Developer/Xcode/XcodeToMetalToolchainIndexMapping.plist");
    if mapping.exists() {
        real
    } else {
        out_dir.to_string_lossy().into_owned()
    }
}

fn compile_metal_shaders(
    shaders_dir: &Path,
    out_dir: &Path,
    target: &str,
    lib_name: &str,
    toolchain: Option<&Path>,
) {
    let cache_root = out_dir.join(".cache");
    std::fs::create_dir_all(&cache_root).expect("Failed to create shader compiler cache");

    // Determine metal language standard from target
    let std_flag = if target.contains("macos26") {
        "-std=metal4.0"
    } else {
        "-std=metal3.1"
    };

    let metal_files: Vec<_> = std::fs::read_dir(shaders_dir)
        .expect("Failed to read shaders directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "metal" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if metal_files.is_empty() {
        println!("cargo:warning=No .metal files found in {:?}", shaders_dir);
        return;
    }

    // Compile each .metal file to .air (intermediate representation)
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_file = out_dir.join(format!("{}_{}.air", lib_name, stem));

        println!("cargo:rerun-if-changed={}", metal_file.display());

        let output = metal_tool_command(toolchain, "metal", out_dir, &cache_root)
            .args([
                // Metal language standard
                std_flag,
                // Optimization flags
                "-O3",
                // Enable fast math for ML workloads
                "-ffast-math",
                // Target Apple Silicon
                "-target",
                target,
                // Include path for shared headers (both metal/ and metal4/)
                "-I",
                shaders_dir.to_str().unwrap(),
                "-I",
                shaders_dir
                    .parent()
                    .unwrap()
                    .join("metal")
                    .to_str()
                    .unwrap(),
                // Compile to AIR
                "-c",
                metal_file.to_str().unwrap(),
                "-o",
                air_file.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to spawn the Metal compiler");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "Failed to compile Metal shader: {}\n\n--- compiler output ---\n{}",
                metal_file.display(),
                stderr
            );
        }

        air_files.push(air_file);
    }

    // Link all .air files into a single .metallib
    let metallib_file = out_dir.join(format!("{}.metallib", lib_name));

    let mut cmd = metal_tool_command(toolchain, "metallib", out_dir, &cache_root);

    for air_file in &air_files {
        cmd.arg(air_file.to_str().unwrap());
    }

    cmd.args(["-o", metallib_file.to_str().unwrap()]);

    let output = cmd.output().expect("Failed to run metallib linker");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "Failed to link Metal library {}\n\n--- linker output ---\n{}",
            lib_name, stderr
        );
    }

    // Set env var for embedding — uppercase lib name
    let env_name = format!(
        "{}_PATH",
        lib_name.to_uppercase().replace("pmetal_", "PMETAL_")
    );
    println!("cargo:rustc-env={}={}", env_name, metallib_file.display());

    println!(
        "Successfully compiled {} Metal shaders to {:?} (target: {})",
        metal_files.len(),
        metallib_file,
        target
    );
}
