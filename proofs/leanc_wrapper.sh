#!/bin/bash
# leanc_wrapper.sh
# Portably overrides the linker flags for Lean compilation on macOS, while defaulting to normal behavior on Linux/Unix.

# 1. Dynamically locate the 'lean' executable
LEAN_EXE="lean"
if ! command -v lean >/dev/null 2>&1; then
    # Fallback to standard elan location if not in PATH
    if [ -x "$HOME/.elan/bin/lean" ]; then
        LEAN_EXE="$HOME/.elan/bin/lean"
    fi
fi

# 2. Extract toolchain directory and construct path to the real leanc
TOOLCHAIN_DIR=$("$LEAN_EXE" --print-prefix 2>/dev/null)
if [ -n "$TOOLCHAIN_DIR" ] && [ -d "$TOOLCHAIN_DIR" ]; then
    LEANC_PATH="${TOOLCHAIN_DIR}/bin/leanc"
else
    LEANC_PATH="leanc"
fi

# 3. Detect if the invocation is for compiling only (-c) or pre-processing
is_linking=1
for arg in "$@"; do
    if [ "$arg" = "-c" ] || [ "$arg" = "-M" ] || [ "$arg" = "-MM" ] || [ "$arg" = "-E" ] || [ "$arg" = "-S" ]; then
        is_linking=0
        break
    fi
done

# 4. Unset LEAN_CC to avoid infinite recursion
unset LEAN_CC

# 5. Check OS and execute with appropriate flags
IS_MAC=0
if [ "$(uname)" = "Darwin" ]; then
    IS_MAC=1
fi

if [ "$is_linking" -eq 1 ] && [ "$IS_MAC" -eq 1 ]; then
    exec "$LEANC_PATH" "$@" -Wl,-no_data_const
else
    exec "$LEANC_PATH" "$@"
fi
