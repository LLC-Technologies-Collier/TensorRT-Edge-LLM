#!/bin/bash
# Script to run specific Perl tests with local library paths correctly set for TensorRT-Edge-LLM.

if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 <test1.t> [test2.t ...]"
    echo "Error: A list of .t files is required as arguments."
    exit 1
fi

echo "--- Building TensorRT::Edge::LLM Perl Bindings ---"
make || exit 1
echo "--- Build finished ---"

# We use find to locate the correct libasan.so for the current architecture
ASAN_LIB=$(find /usr/lib -name "libasan.so.8" | head -n 1)

if [[ -n "$ASAN_LIB" ]]; then
    export LD_PRELOAD="$ASAN_LIB"
    export ASAN_OPTIONS="detect_leaks=0:abort_on_error=1:detect_odr_violation=0"
fi

export LD_LIBRARY_PATH=".:../build:$LD_LIBRARY_PATH"

# -Mblib ensures we use the compiled XS in blib/
# -It/lib for test helper modules
# -b for blib
prove -Mblib -It/lib -bv "$@"