# Compiler flags shared by the production build (build_overlap.sh) and the tests
# (tests/run_small_checks.sh), so the tests always exercise the production move set.
# SEGMENT_EXCHANGE_MAX_ATTEMPTS=1: the segment-exchange move draws once. The inherited
# default previously used 64 retries, which conditions the proposal and violates detailed balance.
OVERLAP_BASE_FLAGS=(-std=c++17 -DOVERLAP_MODE -DSEGMENT_EXCHANGE_MAX_ATTEMPTS=1 -DMBAR_FIXED_STRIDE_FACTOR=1)
