# Audio_ALG/pipelines/Makefile
#
# AEC(linear) -> echo-aware NR -> RES example programs in two builds of the same algorithm
# (DSP chain: see mono_aec_nr_res/main.c):
#   aec_nr_pipeline         heap-allocated modules
#   aec_nr_pipeline_static  one caller pool, no malloc after init
# Both compose lib/aec's Aec (its own RES gain), lib/nr's denoiser and the synthesis themselves
# from the library calls. The mono DFN2 and Align-ULCNet programs compose the same way
# (dfn/dfn_hop.h, ulcnet/ulcnet_nn_stage.h). All link against the submodule libs (lib/aec,
# lib/nr) and audiolib's FFT/fast_math layer (../../audiolib/common). The only archive built
# here is libaudio_pipeline_dfn.a (the DFN2 stage and rate bridge).
#
# Build isolation: this repo consumes three producer archives (audiolib's libaudio_common.a,
# lib/aec's libaec.a, lib/nr's libanr.a), each CFG_SIG-keyed.
#   Phase 1 (COMMON_LIB unset): a goal that needs a producer archive queries, in order,
#     audiolib's, lib/aec's and lib/nr's own `print-lib-path` (forwarding BACKEND/WERROR/CC/
#     CXX/EXTRA_CFLAGS/NO_STDIO and, for lib/aec and lib/nr, the resolved COMMON_DIR/
#     COMMON_LIB) and re-invokes itself with COMMON_LIB/AEC_LIB/NR_LIB set.
#   Phase 2 (COMMON_LIB known): the real rules. Each producer archive is a FORCE-recipe
#     target and a normal prerequisite of every rule that links it, so relinking happens iff
#     a producer archive's mtime advanced.
# Objects and binaries are keyed obj|bin/<backend>-<cfg_sig>/. CFG_SIG folds in this
# Makefile's flag coverage (CC/CXX/AR/RANLIB/CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/
# EXTRA_LDFLAGS/BACKEND/WERROR/NO_STDIO/DEBUG) plus the three producers' config identities
# (COMMON_PRODUCER_SIG/AEC_PRODUCER_SIG/NR_PRODUCER_SIG, the last path component of each
# resolved archive dir), so a differently configured producer gets its own directory.
#
# Usage:
#   make                       # all three producer archives + both pipeline binaries
#   make EXTRA_CFLAGS="..."    # forwarded to the producers and this compile
#   make WERROR=1              # -Werror for this file's own TUs
#   make test                  # build + run the mono chain, board-adapter and stage tests
#   make NO_STDIO=1 libaudio_pipeline_dfn.a   # stdio-less archive only (no link, so no
#                              # producer archive is needed)
#   make audit-no-stdio        # PASS/FAIL: the NO_STDIO=1 libaudio_pipeline_dfn.a has no
#                              # fprintf/printf/puts/fputs/stderr refs and a keyed path that
#                              # differs from the default (stdio) config's
#   make print-bin-dir | print-obj-dir | print-lib-path
#                              # this invocation's own keyed paths
#   make publish               # copy this build's artifacts to dist/<backend>/<cfg_sig>/ and
#                              # update dist/<backend>/current
#   make libs                  # just the three producer archives
#   make clean                 # remove this dir's obj/bin trees
#   make clean-all             # also clean lib/aec, lib/nr and audiolib (clean-libs: alias)
#
# The four-channel pipeline has an independent producer set in 4ch_aec_bf_nr_res/Makefile, so
# its DOA/BF dependencies never change this mono pipeline's build identity.

# Whitespace build-root fail-fast guard: resolved directory paths (COMMON_DIR, transitively AEC_DIR/
# NR_DIR) are forwarded to sub-makes as bare, unquoted words on recipe lines. A space in the path
# would split them into stray arguments and fail deep inside an unrelated sub-invocation, so this
# check, placed before COMMON_DIR/BACKEND/CC/CXX get defaults and before any recipe runs, fails at
# once. $(CURDIR) is the build root they resolve against. A literal space has no token form in Make
# function syntax, so it is built from an empty variable doubled around one.
AUDIO_PIPELINE_EMPTY :=
AUDIO_PIPELINE_SPACE := $(AUDIO_PIPELINE_EMPTY) $(AUDIO_PIPELINE_EMPTY)
ifneq ($(findstring $(AUDIO_PIPELINE_SPACE),$(CURDIR)),)
$(error this build root path contains whitespace ('$(CURDIR)'), which is not supported -- re-run from a path with no spaces. (Whitespace build-root paths are a known, pre-existing limitation shared by all four repos in this stack -- audiolib, lib/aec, lib/nr, and this repo -- not something this pass fixes; this check only fails fast and clearly, here, instead of a cryptic "No such file or directory" deep inside an unrelated sub-invocation.))
endif

AEC_DIR = ../lib/aec/c_impl
NR_DIR  = ../lib/nr/c_impl

# Align-ULCNet C pre/post (STFT/WOLA + UlcnetModel callback boundary) --
# consumed by the mono Align-ULCNet example and ulcnet/ulcnet_nn_stage.c below. Header-only
# include path plus one TU (ulcnet_process.c) compiled by THIS Makefile (that file
# is not in any producer archive). Not part of CFG_SIG's caller-varying
# axes: a fixed repo-relative constant, like AEC_DIR/NR_DIR above.
AIAEC_DIR = ../AIAEC
ULCNET_DIR = $(AIAEC_DIR)/Align_ULCNet
AINR_DIR = ../AINR
DFN2_DIR = $(AINR_DIR)/DeepFilterNet2

# Align-ULCNet deployment grid. One binary serves one grid: the C structs
# embed the STFT/WOLA tables and scratch, so the grid is a build parameter,
# not a runtime field. Selected by name here rather than by hand-writing the
# two -D flags into EXTRA_CFLAGS, because EXTRA_CFLAGS is forwarded verbatim
# to the AEC/NR/audiolib/common producer sub-makes -- none of which contain any
# ULCNet code -- and would split their object caches for no reason.
# Applied ONLY to the ULCNet TUs below, and folded into CFG_SIG_PAYLOAD so
# the two grids never share an object directory.
ULCNET_GRID ?= 16k
ifeq ($(ULCNET_GRID),16k)
ULCNET_GRID_CFLAGS = -DULCNET_MODEL_IO_SR=16000 -DULCNET_MODEL_IO_N_FFT=512
else ifeq ($(ULCNET_GRID),48k)
ULCNET_GRID_CFLAGS = -DULCNET_MODEL_IO_SR=48000 -DULCNET_MODEL_IO_N_FFT=1024
else
$(error ULCNET_GRID must be 16k or 48k, got '$(ULCNET_GRID)')
endif

# DeepFilterNet2/3 lookahead pair (AINR/README.md "Lookahead pairs"): mask 0..2, deep filter
# 0..DF_ORDER-1 (4), plain decimals. A pair other than the headers' default 1/1 becomes -D flags
# on every unit this Makefile compiles (the DFN2/DFN3 sources and everything including their
# headers), so the shipped pair adds no flag and a default build's compile lines are unchanged.
# The pair is folded into CFG_SIG_PAYLOAD: another pair gets its own obj/bin directories, and
# objects of different pairs cannot link (the AINR link guard).
DFN2_MASK_LOOKAHEAD ?= 1
DFN2_DF_LOOKAHEAD ?= 1
DFN3_MASK_LOOKAHEAD ?= 1
DFN3_DF_LOOKAHEAD ?= 1
$(foreach v,DFN2_MASK_LOOKAHEAD DFN3_MASK_LOOKAHEAD,\
  $(if $(filter-out 1,$(words $($(v))))$(filter-out 0 1 2,$($(v))),\
    $(error $(v) must be 0, 1 or 2, got '$($(v))')))
$(foreach v,DFN2_DF_LOOKAHEAD DFN3_DF_LOOKAHEAD,\
  $(if $(filter-out 1,$(words $($(v))))$(filter-out 0 1 2 3 4,$($(v))),\
    $(error $(v) must be 0..4 (DF_ORDER - 1), got '$($(v))')))
DFN_PAIR_CFLAGS :=
ifneq ($(DFN2_MASK_LOOKAHEAD)/$(DFN2_DF_LOOKAHEAD),1/1)
DFN_PAIR_CFLAGS += -DDFN2_MASK_LOOKAHEAD=$(DFN2_MASK_LOOKAHEAD) -DDFN2_DF_LOOKAHEAD=$(DFN2_DF_LOOKAHEAD)
endif
ifneq ($(DFN3_MASK_LOOKAHEAD)/$(DFN3_DF_LOOKAHEAD),1/1)
DFN_PAIR_CFLAGS += -DDFN3_MASK_LOOKAHEAD=$(DFN3_MASK_LOOKAHEAD) -DDFN3_DF_LOOKAHEAD=$(DFN3_DF_LOOKAHEAD)
endif

# audiolib sits two levels up from pipelines/ (SE/Audio_ALG/pipelines -> SE/audiolib/common).
# lib/aec/c_impl and lib/nr/c_impl resolve it themselves from their own location; COMMON_DIR here
# feeds this Makefile's CFLAGS/-I path and is passed to the lib/aec and lib/nr sub-makes as
# COMMON_DIR=$(abspath $(COMMON_DIR)) so all three use the same audiolib checkout.
COMMON_DIR ?= ../../audiolib/common
# audiolib repo root (COMMON_DIR's parent): check-pins reports the whole
# audiolib repo's HEAD, not just the common/ subdirectory's.
AUDIOLIB_DIR ?= ../../audiolib
# ne10 is the only FFT backend. BACKEND stays a variable because it names the
# keyed obj/bin directories and is forwarded to the producers.
BACKEND ?= ne10
ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): ne10 is the only FFT backend)
endif

# Announce the resolved backend on every build (same MAKECMDGOALS guard as
# audiolib's own banner: print-bin-dir/print-obj-dir/print-lib-path are
# consumed via `$$(make -s ... print-lib-path)` command substitution
# elsewhere, and this must not add stray stdout to that captured value).
ifeq ($(filter print-%,$(MAKECMDGOALS)),)
$(info Audio_ALG/pipelines: FFT backend = $(BACKEND))
endif

CC  ?= gcc
CXX ?= c++
AR      ?= ar
RANLIB  ?= ranlib

# CXXFLAGS/CPPFLAGS are not consumed by this Makefile (no C++ TU of its own), but the CFG_SIG
# payload includes them so its shape matches audiolib's/AEC's/NR's and a later use keys builds
# correctly. They are also guarded by the command-line override rejection below. `?=` is a no-op
# when the environment already defines the variable, so its origin would never become
# "environment override" under `make -e`; `+=` is an unconditional assignment attempt that does
# flip it (so the foreach rejects it) while appending an empty string leaves the value unchanged,
# so a plain environment value still folds in. EXTRA_LDFLAGS/NO_STDIO are real here (see
# LDFLAGS and the NO_STDIO note below).
CXXFLAGS      +=
CPPFLAGS      +=
EXTRA_LDFLAGS ?=
NO_STDIO      ?= 0
DEBUG         ?= 0

# DEBUG has no DEBUG-gated code path in this Makefile, but it is forwarded through $(FWD) to every
# producer sub-make, and lib/nr's rejects any non-0/1 value. Validating it here, with an exact
# match (`ifneq` twice, not a single $(filter DEBUG,0 1), so `DEBUG='0 1'` or a value containing
# `%` cannot slip through), stops a stray exported DEBUG from silently building the wrong config
# downstream or failing deep inside a producer sub-build.
ifneq ($(DEBUG),0)
ifneq ($(DEBUG),1)
$(error DEBUG must be exactly 0 or 1 (got DEBUG='$(DEBUG)') -- pass DEBUG=0 \
  (default) or DEBUG=1. Any other value -- including one inherited from a \
  stray exported DEBUG environment variable -- used to silently pass \
  through to producer sub-makes (e.g. lib/nr's, which now rejects it) \
  instead of erroring here at the point Audio_ALG itself is invoked; it no \
  longer does)
endif
endif

# TOOLCHAIN_CHECK=0 skips the CC/CXX -dumpmachine target-coherence guard in
# _cfg_guard below. Forwarded to all three producer
# sub-makes via $(FWD); participates in CFG_SIG.
TOOLCHAIN_CHECK ?= 1

# DIST_ROOT / HOSTCC: where `make publish` writes releases (test scripts point DIST_ROOT at a
# throwaway directory) and the HOST compiler for the publish-time atomic symlink-swap helper
# (never $(CC): that may be a cross compiler). Neither affects what is built, so neither is in
# CFG_SIG. See audiolib's Makefile.

# ALLOW_DIRTY_PUBLISH: `make publish` refuses by default when this repo or any producer
# (audiolib, lib/aec, lib/nr) has uncommitted TRACKED changes (`git status --porcelain -uno`,
# `git -C <dir>` for producers) or is not a git checkout. A checkout with no git identity is
# refused unconditionally, as provenance needs a commit OID. =1 is the escape hatch for dev
# publishes: the attestation records it with a sha256 of each dirty repo's `git diff --binary HEAD`.
# It covers tracked changes only.

# ALLOW_UNTRACKED_PUBLISH: publish also refuses by default when this repo (outside the whitelist
# below) or any producer has an untracked, non-ignored file (`git ls-files --others
# --exclude-standard`), since such a source can change the artifact without a trace in the tracked
# diff. =1 is the separate escape hatch: it records untracked_tree_sha256 per repo with untracked=1,
# computed by hash_untracked() at the top of _publish_impl over sorted fixed-field records ("L
# <sha256(path)> <sha256(readlink)>" for a symlink, "F <mode> <sha256(path)> <sha256(content)>" for
# a regular file; every variable-length value is hashed so records concatenate unambiguously). A
# path that is neither, or whose stat/readlink/shasum fails, is always refused, naming path and repo.

# This repo's untracked list is whitelisted by the prefix `^AINR/(GTCRN|gtcrn_github)/`: those
# directories are user-owned and never staged, and would otherwise refuse (or, under
# ALLOW_UNTRACKED_PUBLISH=1, perturb the provenance hash of) an otherwise identical publish.
# `gtcrn_github/` is an embedded git checkout that shows as one opaque line in
# `git ls-files --others`, which is also the unhashable case above. The producers get no
# whitelist. Neither knob is in CFG_SIG: they only decide whether publish proceeds.

# ATTEST_STAMP (test only): overrides the UTC timestamp in the attestation filename so the
# isolation tests can force same-second collisions and exercise the no-clobber retry. Attestations
# are unsigned (traceability, not authenticity). Not in CFG_SIG or MANIFEST.
DIST_ROOT ?= dist
HOSTCC    ?= cc
ALLOW_DIRTY_PUBLISH ?= 0
ALLOW_UNTRACKED_PUBLISH ?= 0
ATTEST_STAMP ?=

# EXTRA_CFLAGS is forwarded to the producer builds and to this pipeline's own compile (the
# producers' Makefiles already do `CFLAGS += $(EXTRA_CFLAGS)`). The matched-filter and
# delay-estimator duty-cycle arithmetic is built in unconditionally (lib/aec/c_impl/src/delay_aec.c).
# Changing EXTRA_CFLAGS, SIMD, WERROR or NO_STDIO needs no `make clean-libs`: every Makefile in the
# stack keys its objects/archives on the same axes.
EXTRA_CFLAGS ?=
# PROFILE=1 turns on lib/aec's per-hop stage timing (the knob is forwarded by name). Off by
# default so a release build takes no clock reads. Folded into CFG_SIG by name and through CFLAGS,
# so profile and release builds occupy different obj/bin directories.
PROFILE ?= 0
ifneq ($(words $(PROFILE)),1)
$(error PROFILE must be exactly 0 or 1, got '$(PROFILE)')
endif
ifneq ($(filter-out 0 1,$(PROFILE)),)
$(error PROFILE must be exactly 0 or 1, got '$(PROFILE)')
endif
# PROFILE_CFLAGS goes into CFLAGS rather than EXTRA_CFLAGS, as SIMD does below: EXTRA_CFLAGS is
# forwarded to the producer sub-makes and folded into CFG_SIG, so a -D smuggled through it arrived
# twice when a consumer with its own PROFILE drove this build, splitting identical objects across
# two keys and re-keying producers that never use the macro. CFLAGS is protected from command-line
# assignment (FP policy guard below). These -D land in this Makefile's CFLAGS only: lib/aec gets
# AEC_STAGE_TIMING from the PROFILE knob forwarded in FWD.

# AEC_STAGE_TIMING is repeated on purpose: this layer's own TUs test it, to decide what the tests
# may assert about the AEC record. A lib/aec pin without the PROFILE knob leaves the AEC block at
# zero.
ifeq ($(PROFILE),1)
PROFILE_CFLAGS := -DAEC_STAGE_TIMING=1
else
PROFILE_CFLAGS :=
endif

SIMD ?= 1
ifneq ($(words $(SIMD)),1)
$(error SIMD must be exactly 0 or 1, got '$(SIMD)')
endif
ifneq ($(filter-out 0 1,$(SIMD)),)
$(error SIMD must be exactly 0 or 1, got '$(SIMD)')
endif
ifeq ($(SIMD),0)
SIMD_CFLAGS := -DSIMD_KERNELS_FORCE_SCALAR
else
SIMD_CFLAGS :=
endif

# -ffp-contract=off applies to every pipeline TU: contraction of mul+add pairs would change bytes,
# and one uniform flag keeps new SIMD/EMA code under the repo-wide bit-exactness discipline. It is
# appended at the true end of the CFLAGS assembly (see "FP-contraction policy" below), so a caller's
# EXTRA_CFLAGS can never land after it and override it.
CFLAGS = -Wall -Wextra -O2 -std=gnu99 \
         -I. \
         -I$(ULCNET_DIR) -I$(DFN2_DIR) \
         -I$(AEC_DIR)/include -I$(AEC_DIR)/example \
         -I$(NR_DIR)/include \
         -I$(COMMON_DIR)/include -I$(COMMON_DIR)
# AUDIO_PIPELINE_BACKEND_STR: compile-time FFT backend identity baked into the DFN stage and
# rate-bridge descriptors (backend_id); it matches the libaudio_common.a variant this build links
# (BACKEND= above), so the descriptor never needs the backend told separately.
CFLAGS += -DAUDIO_PIPELINE_BACKEND_STR=\"$(BACKEND)\"
ifneq ($(DFN_PAIR_CFLAGS),)
CFLAGS += $(DFN_PAIR_CFLAGS)
endif
CFLAGS += $(EXTRA_CFLAGS)
CFLAGS += $(SIMD_CFLAGS)
CFLAGS += $(PROFILE_CFLAGS)

# -Werror opt-in for this Makefile's own TUs (default off; CI can set WERROR=1). Vendor code
# (audiolib's NE10) is built by audiolib's own Makefile, so no separate warning isolation is needed.
WERROR ?= 0
ifeq ($(WERROR),1)
  CFLAGS += -Werror
endif

# NO_STDIO=1 is forwarded to the producers (audiolib, lib/aec, lib/nr compile their diagnostics
# out) and folded into CFG_SIG; this repo's archived objects use no stdio at all. The host
# programs and the tests use stdio unconditionally.

# --- FP-contraction policy (unified across the four repos) -------------------
# -ffp-contract=off is appended LAST to CFLAGS (`CFLAGS += $(FP_POLICY)` below), after
# EXTRA_CFLAGS/WERROR and before CFG_SIG_PAYLOAD, so a caller's EXTRA_CFLAGS=-ffp-contract=fast
# (or -Ofast/-ffast-math) cannot land later and win.
# See audiolib's Makefile and its scripts/audit_fp_contract.sh for the design and the
# disassembly-level check.

# FP_POLICY is a bare `:=` literal defined here, ahead of the override-rejection foreach below,
# because `$(origin FP_POLICY)` only becomes "environment override" once an assignment has been
# parsed: defined lower, `env FP_POLICY=-ffp-contract=fast make -e` would pass unchecked and drop
# -ffp-contract=off. -fno-math-errno lets audiolib's AArch64 sqrtf lower to FSQRT without an
# errno-checking libm fallback; scalar and NEON operation ordering stays stable.
FP_POLICY := -ffp-contract=off -fno-math-errno

# FP_ALLOWED_CHARS_RE (the allow-list's character set, as in audiolib/AEC/NR) is a second bare `:=`
# literal with the same `make -e` exposure as FP_POLICY, so it is defined early. Its regex is
# documented where it is used (the allow-list block below).
FP_ALLOWED_CHARS_RE := ^[A-Za-z0-9_./=,+ -]*$$

# LDFLAGS is link flags only (-lm plus EXTRA_LDFLAGS), not the producer archive paths: those are
# resolved by the two-phase dispatch below and appended to each link recipe in the fixed order
# program objects, libaudio_pipeline_dfn.a (where used), libaec.a, libanr.a, libaudio_common.a,
# -lm. Defined ahead of the override-rejection foreach for the same `make -e` origin reason as
# FP_POLICY.
LDFLAGS = -lm
LDFLAGS += $(EXTRA_LDFLAGS)

# --- Command-line override rejection ------------------------------------
# A command-line CFLAGS=/CXXFLAGS=/CPPFLAGS=/LDFLAGS=/FP_POLICY=/FP_ALLOWED_CHARS_RE= silently
# disables this file's own appends to it (the FP policy, the backend define, the -I paths, -lm), and
# an override of FP_ALLOWED_CHARS_RE could widen the allow-list below to match anything.
# EXTRA_CFLAGS / EXTRA_LDFLAGS are the supported hooks. See audiolib's Makefile.
$(foreach v,CFLAGS CXXFLAGS CPPFLAGS LDFLAGS FP_POLICY FP_ALLOWED_CHARS_RE,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy, backend define, include paths); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead)))

# Conflict detection and the quote/escape/response-file rejection gate (as in audiolib/AEC/NR):
# reject outright rather than let a caller re-enable contraction. It checks the fully assembled
# CFLAGS/CXXFLAGS/CPPFLAGS (EXTRA_CFLAGS and any command-line or environment value already folded
# in) before FP_POLICY is appended and before CFG_SIG is computed, so a rejected build never
# creates an obj/bin directory. All three variables are inspected: a plain-environment
# CXXFLAGS=-Ofast would pass a CFLAGS-only check.

# Matching is by whole word ($(filter), with `%` for -ffp-contract=<x>), not substring
# ($(findstring)): a harmless `-DNOTE=-Ofastness` is not the flag -Ofast.

# FP_INPUT_FLAGS excludes the one Makefile-generated token `-DAUDIO_PIPELINE_BACKEND_STR=%`, which
# legitimately contains a backslash and double quotes (dfn_res_stage.c reads it as a C string
# literal). Without the exclusion an override-free `make all` would trip the quote/allow-list checks
# on this file's own CFLAGS. The token has a fixed shape and is never caller-controlled, so it
# cannot carry a conflicting flag past the gate. The same FP_INPUT_FLAGS feeds the single-quote
# check, the allow-list $(shell) call and FP_CONFLICT_FLAGS.
FP_INPUT_FLAGS := $(filter-out -DAUDIO_PIPELINE_BACKEND_STR=%,$(CFLAGS) $(CXXFLAGS) $(CPPFLAGS))

# --- link-flags character-safety coverage (as in audiolib's) ------------------
# The allow-list also validates LDFLAGS (-lm plus EXTRA_LDFLAGS), because LDFLAGS is embedded in
# every link recipe just as CFLAGS is in every compile recipe: `EXTRA_LDFLAGS=';id'` would otherwise
# reach a real `cc -o ... $(LDFLAGS)` line live. The two character-safety checks therefore run on
# SHELL_SAFE_INPUT_FLAGS (FP_INPUT_FLAGS plus LDFLAGS), sharing one $(shell) call. The FP-semantic
# check (FP_CONFLICT_FLAGS) stays on FP_INPUT_FLAGS only: -Ofast/-ffast-math/-ffp-contract are
# meaningless as link flags.
SHELL_SAFE_INPUT_FLAGS := $(FP_INPUT_FLAGS) $(LDFLAGS)

# Make compares words of its own text and has no notion of shell quoting: EXTRA_CFLAGS="'-Ofast'",
# '"-ffast-math"' and "-O'f'ast" are stored with their quote characters, so $(filter) cannot match
# them, yet the shell strips the quotes and the compiler receives -Ofast. An @response-file token is
# equally invisible to a whole-word check.

# A character deny-list cannot keep up: globs, `~` and above all the redirections `>` `<` and
# process substitution are not Make syntax, so Make neither resolves nor rejects them. The input is
# therefore checked as a whole against one ALLOW-list with `grep -E`, a $(shell) call like CFG_SIG's
# cksum below. SHELL_SAFE_INPUT_FLAGS is embedded single-quoted in that command, where every
# character but a single quote is inert; so the single-quote check below stays a separate
# pure-$(findstring) pre-check (no shell involved) that must run first.
ifneq ($(findstring ',$(SHELL_SAFE_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS contains a single-quote character; this repo requires plain, unquoted, whitespace-separated compiler/linker argv tokens built only from the allowed character set (see the allow-list below) -- a single quote specifically must be rejected by this Make-native check BEFORE the allow-list's own $(shell) call can safely single-quote-embed the rest of the text)
endif

# The allow-list: letters, digits, underscore, dot, slash, equals, comma, plus and minus, plus
# space as the token separator. Every flag this Makefile uses is built from this set; comma and plus
# stay for -Wl,-rpath,dir / -Wa,--option style pass-through and are inert to the shell. LC_ALL=C pins
# the bracket ranges to ASCII ordering on BSD and GNU grep/tr. Every deny-list character (quotes,
# backslash, backtick, "$(", `;`, `|`, `&`, @-response-file) is outside the set and so rejected by
# construction; scripts/test_build_isolation.sh covers it. FP_ALLOWED_CHARS_RE is defined near the top.
SHELL_SAFE_ALLOWLIST_RC := $(shell printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | LC_ALL=C grep -Eq '$(FP_ALLOWED_CHARS_RE)'; echo $$?)
ifneq ($(SHELL_SAFE_ALLOWLIST_RC),0)
# Diagnostic-only second $(shell) call (runs only on the error path): it deletes every allowed
# character from the same text and dedupes the rest, so the $(error ...) below names the specific
# offending characters (e.g. ">" or "~").
SHELL_SAFE_DISALLOWED_CHARS := $(shell printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | LC_ALL=C tr -d 'A-Za-z0-9_./=,+ -' | fold -w1 | LC_ALL=C sort -u | tr -d '\n')
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS contains a character outside the allowed set [A-Za-z0-9_./=,+- ] (letters/digits/underscore/dot/slash/equals/comma/plus/minus, plus space as the token separator) -- disallowed character(s) found: "$(SHELL_SAFE_DISALLOWED_CHARS)"; this repo requires plain, unquoted, whitespace-separated compiler/linker argv tokens built only from that set -- remove/replace the disallowed character(s) shown above in CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS)
endif

FP_CONFLICT_FLAGS := $(filter -Ofast -ffast-math -ffp-contract=%,$(FP_INPUT_FLAGS))
ifneq ($(FP_CONFLICT_FLAGS),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains $(FP_CONFLICT_FLAGS); this repo pins -ffp-contract=off; remove $(FP_CONFLICT_FLAGS) from CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS)
endif

# Override rejection for the four internal variables of the checks above (FP_INPUT_FLAGS,
# SHELL_SAFE_INPUT_FLAGS, SHELL_SAFE_ALLOWLIST_RC, FP_CONFLICT_FLAGS): a command-line value would win
# over the computed one and turn every gate above into a no-op while the real CFLAGS/LDFLAGS stay
# unvalidated. They are computed, not bare literals, so this foreach must run after their last
# assignment (so $(origin) has already flipped) instead of joining the early foreach.
# SHELL_SAFE_DISALLOWED_CHARS is excluded: it is only diagnostic text on the failure path.
$(foreach v,FP_INPUT_FLAGS SHELL_SAFE_INPUT_FLAGS SHELL_SAFE_ALLOWLIST_RC FP_CONFLICT_FLAGS,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently defeat the FP-policy/shell-safety allow-list/conflict checks above, leaving the real CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS content unvalidated; this variable is internal to those checks and has no supported external use)))

# FP_POLICY's `:=` literal is defined earlier (see "FP-contraction policy"); only its CFLAGS append,
# which must come last, lives here.
CFLAGS     += $(FP_POLICY)

# The NE10 archive carries one C++ TU, so plain-C binaries link with the C++ driver, which supplies
# the right C++ runtime itself (libc++ on macOS/clang, libstdc++ on GNU/Linux gcc); a hardcoded
# `-lc++` would be a link error on GNU/Linux. LINK participates in CFG_SIG below.
LINK = $(CXX)

.DEFAULT_GOAL := all

# Consumer two-phase, THREE-producer resolution: all three producers' bin/ directories are
# CFG_SIG-keyed, so there is no fixed archive path to hardcode, and a producer built with a different
# CC/CXX/NO_STDIO must show in this repo's build identity.
#   Phase 1 (COMMON_LIB unset): resolve audiolib's archive path first (lib/aec's and lib/nr's own
#     dispatch need it), then lib/aec's, then lib/nr's, and re-invoke with all three set.
#   Phase 2 (COMMON_LIB known): the real rules, with $(COMMON_LIB)/$(AEC_LIB)/$(NR_LIB) as normal
#     prerequisites of the binaries that link them, so relinking happens iff an archive's mtime advanced.

# clean/clean-all have real rules outside the dispatch; libaudio_pipeline_dfn.a and the audit
# targets go through it so the archive's keyed path folds in the producer identities.

# `make -n`: the dispatch recipe mentions $(MAKE), so GNU Make runs it even under -n; -n reaches each
# child make through MAKEFLAGS, so the children only print their recipes and create no files.

# lib/aec's and lib/nr's CFG_SIG include COMMON_LIB (as COMMON_PRODUCER_SIG) for every target, even
# their `lib` target, which never links audiolib. COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB=<path>
# must therefore reach every lib/aec and lib/nr sub-make (queries and real builds): otherwise a `lib`
# build would land in a different keyed directory than the query resolved and the archive-exists
# assertion in the FORCE rule would fail.
FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) PROFILE=$(PROFILE) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) DEBUG=$(DEBUG)

ifeq ($(COMMON_LIB),)
.PHONY: all test test_target_abi target-abi-test libs _publish_impl aec_nr_pipeline aec_nr_pipeline_static mono_alignulcnet mono_aec_dfn_res test_mono_chain test_mono_ulcnet test_ulcnet_accelerator_adapter test_dfn_res_stage test_dfn_rate_bridge test_mono_dfn test_mcab_sample_post test_readme_flow test_dfn_hop test_ulcnet_nn_stage example-adapter libaudio_pipeline_dfn.a
all test test_target_abi target-abi-test libs _publish_impl aec_nr_pipeline aec_nr_pipeline_static mono_alignulcnet mono_aec_dfn_res test_mono_chain test_mono_ulcnet test_ulcnet_accelerator_adapter test_dfn_res_stage test_dfn_rate_bridge test_mono_dfn test_mcab_sample_post test_readme_flow test_dfn_hop test_ulcnet_nn_stage example-adapter libaudio_pipeline_dfn.a:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audiolib archive path (COMMON_DIR=$(COMMON_DIR) -- is audiolib present as a sibling of Audio_ALG?)" >&2; exit 1; }; \
	 aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 test -n "$$aec" || { echo "FATAL: could not resolve lib/aec archive path (AEC_DIR=$(AEC_DIR))" >&2; exit 1; }; \
	 nr="$$($(MAKE) -s --no-print-directory -C $(NR_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 test -n "$$nr" || { echo "FATAL: could not resolve lib/nr archive path (NR_DIR=$(NR_DIR))" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory $@ COMMON_LIB="$$ac" AEC_LIB="$$aec" NR_LIB="$$nr"
else
# Producer-identity components of this consumer's CFG_SIG: the last path component of each
# resolved archive's directory (e.g. "ne10-2139547423"), never the full absolute path, so a pipeline
# build against a differently configured producer keys to its own obj/bin directory. COMMON_DIR/
# AEC_DIR/NR_DIR (abspaths, used only as -C arguments) never appear in a payload or manifest.
COMMON_PRODUCER_SIG  := $(notdir $(patsubst %/,%,$(dir $(COMMON_LIB))))
AEC_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(AEC_LIB))))
NR_PRODUCER_SIG  := $(notdir $(patsubst %/,%,$(dir $(NR_LIB))))
endif

# --- Hash-keyed object/binary directory + full-coverage CFG_SIG ------------
# obj|bin/<backend>-<sig>, as in audiolib's Makefile. The payload covers CC CXX AR RANLIB CFLAGS
# CXXFLAGS CPPFLAGS LDFLAGS EXTRA_CFLAGS EXTRA_LDFLAGS BACKEND WERROR NO_STDIO DEBUG, the three
# producer identities (empty while a _LIB is unresolved), LINK, TOOLCHAIN_CHECK (a guard-skip knob
# must not alias a guarded build) and SRCS. It is defined after CFLAGS/LDFLAGS are final and the
# PRODUCER_SIGs are resolved, but before the first rule that mentions $(OBJ_DIR)/$(BIN_DIR): Make
# expands target lists when parsed, so a forward reference would see an empty CFG_SIG. Not $(strip)'d
# (a false miss is fine, a false hit never); cksum (POSIX) hashes it.

# One optional descriptor test (test_target_abi) spans all public pipeline/prepost APIs.
TARGET_ABI_SRCS := tests/test_target_abi.c \
 4ch_aec_bf_nr_res/mcab.c \
 dfn/dfn_res_stage.c dfn/dfn_rate_bridge.c \
 $(DFN2_DIR)/dfn2_process.c $(DFN2_DIR)/dfn2_model_io.c $(DFN2_DIR)/dfn2_prepost.c \
 ../AINR/DeepFilterNet3/dfn3_process.c ../AINR/DeepFilterNet3/dfn3_model_io.c \
 ../AINR/DeepFilterNet3/dfn3_prepost.c \
 $(ULCNET_DIR)/ulcnet_process.c $(ULCNET_DIR)/ulcnet_model_io.c $(ULCNET_DIR)/ulcnet_prepost.c \
 $(AIAEC_DIR)/aiaec_process.c $(AIAEC_DIR)/DeepVQE_S/deepvqe_process.c \
 $(AIAEC_DIR)/DeepVQE_S/deepvqe_prepost.c
TARGET_ABI_INCLUDES := -I4ch_aec_bf_nr_res \
 -I$(DFN2_DIR) -I../AINR/DeepFilterNet3 -I$(ULCNET_DIR) \
 -I$(AIAEC_DIR) -I$(AIAEC_DIR)/DeepVQE_S -I$(COMMON_DIR)/utility

# The same CC command can select a different SDK through PATH.
# Include effective pointer/FP/ISA predefines to avoid cross-ABI cache reuse.
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) ULCNET_GRID=$(ULCNET_GRID) DFN2_LOOKAHEAD=$(DFN2_MASK_LOOKAHEAD)/$(DFN2_DF_LOOKAHEAD) DFN3_LOOKAHEAD=$(DFN3_MASK_LOOKAHEAD)/$(DFN3_DF_LOOKAHEAD) CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) LDFLAGS=$(LDFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) SIMD=$(SIMD) PROFILE=$(PROFILE) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) COMMON_PRODUCER_SIG=$(COMMON_PRODUCER_SIG) AEC_PRODUCER_SIG=$(AEC_PRODUCER_SIG) NR_PRODUCER_SIG=$(NR_PRODUCER_SIG) SRCS=$(sort mono_aec_nr_res/main.c mono_aec_nr_res/tests/test_mono_chain.c mono_aec_nr_res/tests/test_no_stdio_stack.c mono_aec_nr_res/example_board_adapter.c ulcnet/ulcnet_nn_stage.c ulcnet/mono_alignulcnet/tests/test_mono_ulcnet.c $(ULCNET_DIR)/ulcnet_accelerator_adapter.c $(ULCNET_DIR)/ulcnet_prepost.c $(ULCNET_DIR)/ulcnet_process.c $(ULCNET_DIR)/ulcnet_model_io.c $(AIAEC_DIR)/tests/test_ulcnet_accelerator_adapter.c ulcnet/mono_alignulcnet/main.c dfn/dfn_res_stage.c dfn/dfn_rate_bridge.c dfn/tests/test_dfn_res_stage.c dfn/tests/test_dfn_rate_bridge.c 4ch_aec_bf_nr_res/tests/test_mcab_sample_post.c tests/test_readme_flow.c dfn/mono_aec_dfn_res/tests/test_mono_dfn.c dfn/mono_aec_dfn_res/main.c dfn/tests/test_dfn_hop.c ulcnet/tests/test_ulcnet_nn_stage.c $(DFN2_DIR)/dfn2_process.c $(DFN2_DIR)/dfn2_model_io.c $(DFN2_DIR)/dfn2_prepost.c $(TARGET_ABI_SRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

# OBJ_ROOT / BIN_ROOT: where this repo's keyed obj/bin trees live (default obj/ and bin/ here). A
# placement knob like DIST_ROOT, so not in CFG_SIG; the isolation tests point them at a scratch
# directory so the real obj/ and bin/ are never touched. Each producer keys its own trees.
OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR  = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR  = $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)

# --- CFG_SIG collision guard + directory creation ---------------------------
# Single order-only prerequisite for every compile/link/archive rule below.
# See audiolib's Makefile for the full design writeup (identical
# mechanism, applied here) -- write config.manifest first time, verify (hard
# error on mismatch = CFG_SIG collision or payload coverage gap) every time
# after, atomically (temp file + mv).
.PHONY: _cfg_guard
_cfg_guard:
	@test -n "$(CC_TARGET_SIG)" && test "$(CC_TARGET_SIG)" != unavailable || { \
	  echo "FATAL: cannot determine compiler target; CC preprocessing with the configured flags must succeed." >&2; \
	  exit 1; }
	@if [ "$(TOOLCHAIN_CHECK)" = "1" ]; then \
	  cc_m="$$($(CC) -dumpmachine 2>/dev/null)"; cxx_m="$$($(CXX) -dumpmachine 2>/dev/null)"; \
	  if [ -z "$$cc_m" ] || [ "$$cc_m" != "$$cxx_m" ]; then \
	    echo "FATAL: CC/CXX compile for different targets (CC='$(CC)' -> '$$cc_m', CXX='$(CXX)' -> '$$cxx_m')." >&2; \
	    echo "  The build links a C++ TU (audiolib's NE10 archive); pass the FULL toolchain:" >&2; \
	    echo "    make CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ AR=aarch64-linux-gnu-ar RANLIB=aarch64-linux-gnu-ranlib" >&2; \
	    echo "  (TOOLCHAIN_CHECK=0 skips this check; it participates in CFG_SIG.)" >&2; \
	    exit 1; \
	  fi; \
	fi
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)
	@for d in $(OBJ_DIR) $(BIN_DIR); do \
	  m="$$d/config.manifest"; \
	  if [ -f "$$m" ]; then \
	    old="$$(cat "$$m")"; \
	    if [ "$$old" != "$(CFG_SIG_PAYLOAD)" ]; then \
	      echo "FATAL: CFG_SIG collision or coverage gap in $$m" >&2; \
	      echo "  CFG_SIG=$(CFG_SIG)" >&2; \
	      echo "  stored : $$old" >&2; \
	      echo "  current: $(CFG_SIG_PAYLOAD)" >&2; \
	      exit 1; \
	    fi; \
	  else \
	    tmp="$$d/.config.manifest.tmp.$$$$"; \
	    printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$tmp"; \
	    mv -f "$$tmp" "$$m"; \
	  fi; \
	done

# -MD -MP: emits $(OBJ_DIR)/<name>.d per object (header dependency tracking)
# -- included near the bottom of this file via `-include $(wildcard $(OBJ_DIR)/*.d)`.
# One source, two binaries: aec_nr_pipeline splits the pool in two (AEC, post stages).
$(OBJ_DIR)/aec_nr_pipeline.o: mono_aec_nr_res/main.c common/example_cli.h | _cfg_guard
	$(CC) $(CFLAGS) -DAEC_NR_POOL_SPLIT=1 -MD -MP -c -o $@ $<

$(OBJ_DIR)/aec_nr_pipeline_static.o: mono_aec_nr_res/main.c common/example_cli.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# test_mono_chain: the mono chain over lib/aec's Aec -- pool carve vs heap, reset, switches,
# comfort noise, chain vs the hand-written oracle (tests/conv_chain_mono.h), and the two example
# programs run as black boxes against the same oracle.
$(OBJ_DIR)/test_mono_chain.o: mono_aec_nr_res/tests/test_mono_chain.c mono_aec_nr_res/tests/conv_chain_mono.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# Align-ULCNet objects. ulcnet_process.o compiles the
# AIAEC-owned C pre/post TU (centered sqrt-Hann STFT / WOLA / model
# boundary) -- read-only from this Makefile's perspective, never edited
# here. It transforms through audiolib's fft_wrapper; the -I$(COMMON_DIR)/
# include already in $(CFLAGS) resolves that header, and -I$(ULCNET_DIR)
# resolves its own. The stage TU and the programs and tests built on it also need the ULCNET
# include path for ulcnet_process.h.
$(OBJ_DIR)/ulcnet_process.o: $(ULCNET_DIR)/ulcnet_process.c $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/ulcnet_model_io.o: $(ULCNET_DIR)/ulcnet_model_io.c $(ULCNET_DIR)/ulcnet_model_io.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

# ulcnet_prepost.o: the AIAEC-owned pre/post class -- one object owning the
# framing states, the feature front end and the per-frame accelerator
# transaction. A THIRD TU on purpose: ulcnet_process.c and ulcnet_model_io.c
# stay independently linkable (each has a parity test that compiles it
# without the other), so the file that composes them cannot be either one.
$(OBJ_DIR)/ulcnet_prepost.o: $(ULCNET_DIR)/ulcnet_prepost.c $(ULCNET_DIR)/ulcnet_prepost.h $(ULCNET_DIR)/ulcnet_process.h $(ULCNET_DIR)/ulcnet_model_io.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/ulcnet_accelerator_adapter.o: $(ULCNET_DIR)/ulcnet_accelerator_adapter.c $(ULCNET_DIR)/ulcnet_accelerator_adapter.h $(ULCNET_DIR)/ulcnet_prepost.h $(ULCNET_DIR)/ulcnet_model_io.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/mono_alignulcnet_main.o: ulcnet/mono_alignulcnet/main.c ulcnet/ulcnet_nn_stage.h common/example_cli.h $(ULCNET_DIR)/ulcnet_accelerator_adapter.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_ulcnet_accelerator_adapter.o: $(AIAEC_DIR)/tests/test_ulcnet_accelerator_adapter.c $(ULCNET_DIR)/ulcnet_accelerator_adapter.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

# ulcnet_nn_stage.o: the Align-ULCNet post-filter stage (ulcnet/ulcnet_nn_stage.h), example
# code linked by the programs and tests that use it. Not a member of any archive.
$(OBJ_DIR)/ulcnet_nn_stage.o: ulcnet/ulcnet_nn_stage.c ulcnet/ulcnet_nn_stage.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

# test_mono_ulcnet: the mono linear AEC feeding the stage (identity end to end, alignment
# events from the real delay state, reprime, model policy on real framing).
$(OBJ_DIR)/test_mono_ulcnet.o: ulcnet/mono_alignulcnet/tests/test_mono_ulcnet.c ulcnet/ulcnet_nn_stage.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

# test_ulcnet_nn_stage: the stage alone (framing, model policy, WOLA) on the compiled grid.
$(OBJ_DIR)/test_ulcnet_nn_stage.o: ulcnet/tests/test_ulcnet_nn_stage.c ulcnet/ulcnet_nn_stage.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn2_process.o: $(DFN2_DIR)/dfn2_process.c $(DFN2_DIR)/dfn2_process.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(DFN2_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn2_model_io.o: $(DFN2_DIR)/dfn2_model_io.c $(DFN2_DIR)/dfn2_model_io.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(DFN2_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn2_prepost.o: $(DFN2_DIR)/dfn2_prepost.c $(DFN2_DIR)/dfn2_prepost.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(DFN2_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn_res_stage.o: dfn/dfn_res_stage.c dfn/dfn_res_stage.h $(DFN2_DIR)/dfn2_prepost.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(DFN2_DIR) -MD -MP -c -o $@ $<

# test_mcab_sample_post: the header-only glue of the four-channel programs; links no mcab object.
$(OBJ_DIR)/test_mcab_sample_post.o: 4ch_aec_bf_nr_res/tests/test_mcab_sample_post.c 4ch_aec_bf_nr_res/mcab_sample_post.h 4ch_aec_bf_nr_res/mcab.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_readme_flow.o: tests/test_readme_flow.c dfn/dfn_hop.h ulcnet/ulcnet_nn_stage.h mono_aec_nr_res/tests/conv_chain_mono.h dfn/tests/dfn_test_fixture.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_dfn_res_stage.o: dfn/tests/test_dfn_res_stage.c dfn/dfn_res_stage.h dfn/tests/dfn_test_fixture.h dfn/dfn_unit_partition.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(DFN2_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn_rate_bridge.o: dfn/dfn_rate_bridge.c dfn/dfn_rate_bridge.h dfn/dfn_res_stage.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(DFN2_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_dfn_rate_bridge.o: dfn/tests/test_dfn_rate_bridge.c dfn/dfn_rate_bridge.h dfn/tests/dfn_test_fixture.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(DFN2_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_mono_dfn.o: dfn/mono_aec_dfn_res/tests/test_mono_dfn.c mono_aec_nr_res/tests/conv_chain_mono.h dfn/dfn_hop.h dfn/dfn_res_stage.h dfn/dfn_rate_bridge.h dfn/tests/dfn_test_fixture.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/mono_aec_dfn_res_main.o: dfn/mono_aec_dfn_res/main.c dfn/dfn_hop.h dfn/dfn_res_stage.h dfn/dfn_rate_bridge.h dfn/dfn_example_args.h dfn/dfn_unit_partition.h common/example_cli.h dfn/dfn_hop_pool.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# test_dfn_hop: the DfnHop paths against direct calls of a twin stage.
$(OBJ_DIR)/test_dfn_hop.o: dfn/tests/test_dfn_hop.c dfn/dfn_hop.h dfn/dfn_res_stage.h dfn/dfn_rate_bridge.h dfn/dfn_unit_partition.h dfn/tests/dfn_test_fixture.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(DFN2_DIR) -MD -MP -c -o $@ $<

# example_board_adapter: REFERENCE ONLY host simulation of a board-side
# common-memory-manager integration -- see that
# file's own header. Not the real board adapter; demonstrates the calling
# sequence/error handling/descriptor contract only.
$(OBJ_DIR)/example_board_adapter.o: mono_aec_nr_res/example_board_adapter.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

DFN_LIB_TARGET = $(BIN_DIR)/libaudio_pipeline_dfn.a
DFN_STAGE_OBJS = $(OBJ_DIR)/dfn_res_stage.o $(OBJ_DIR)/dfn2_prepost.o \
                 $(OBJ_DIR)/dfn2_model_io.o $(OBJ_DIR)/dfn2_process.o
DFN_PIPELINE_OBJS = $(OBJ_DIR)/dfn_rate_bridge.o $(DFN_STAGE_OBJS)

# =============================================================================
# Phase 2 real rules: only defined once COMMON_LIB (and, transitively, AEC_LIB/
# NR_LIB, always set together by phase 1's re-invoke) is known.
# =============================================================================
ifneq ($(COMMON_LIB),)
.PHONY: all test libs aec_nr_pipeline aec_nr_pipeline_static mono_alignulcnet mono_aec_dfn_res test_mono_chain test_mono_ulcnet test_ulcnet_accelerator_adapter test_dfn_res_stage test_dfn_rate_bridge test_mono_dfn test_mcab_sample_post test_readme_flow test_dfn_hop test_ulcnet_nn_stage example-adapter example_board_adapter libaudio_pipeline_dfn.a FORCE
# This branch is entered through the recursive `$(MAKE) ... COMMON_LIB=... AEC_LIB=... NR_LIB=...`
# re-invoke. Make folds command-line variables into MAKEOVERRIDES/MAKEFLAGS for every further
# $(MAKE) recipe, where another Makefile's own producer variables would collide with them.
# Clearing MAKEOVERRIDES stops that without affecting variables forwarded as
# explicit VAR=value recipe text; `unexport` also blocks the separate auto-export of command-line
# variables into recipe environments.
unexport COMMON_LIB AEC_LIB NR_LIB
MAKEOVERRIDES :=
all: $(BIN_DIR)/aec_nr_pipeline $(BIN_DIR)/aec_nr_pipeline_static $(BIN_DIR)/mono_alignulcnet $(DFN_LIB_TARGET) $(BIN_DIR)/mono_aec_dfn_res

libaudio_pipeline_dfn.a: $(DFN_LIB_TARGET)

$(DFN_LIB_TARGET): $(DFN_PIPELINE_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "libaudio_pipeline_dfn.a [$(BACKEND)] -> $@"

# test: also builds and runs example_board_adapter, whose exit code signals any failed positive-flow
# step or negative demonstration. `make example-adapter` (below) runs it standalone.
test: $(BIN_DIR)/test_mono_chain $(BIN_DIR)/aec_nr_pipeline $(BIN_DIR)/aec_nr_pipeline_static $(BIN_DIR)/example_board_adapter $(BIN_DIR)/test_mono_ulcnet $(BIN_DIR)/test_ulcnet_accelerator_adapter $(BIN_DIR)/test_dfn_res_stage $(BIN_DIR)/test_dfn_rate_bridge $(BIN_DIR)/test_mono_dfn $(BIN_DIR)/test_mcab_sample_post $(BIN_DIR)/test_readme_flow $(BIN_DIR)/test_ulcnet_nn_stage $(BIN_DIR)/test_dfn_hop $(BIN_DIR)/mono_alignulcnet $(BIN_DIR)/mono_aec_dfn_res
	"$(abspath $(BIN_DIR))/test_mono_chain" "$(abspath $(BIN_DIR))" "$(abspath $(OBJ_DIR))"
	"$(abspath $(BIN_DIR))/example_board_adapter"
	"$(abspath $(BIN_DIR))/test_mono_ulcnet"
	"$(abspath $(BIN_DIR))/test_ulcnet_accelerator_adapter"
	"$(abspath $(BIN_DIR))/test_dfn_res_stage"
	"$(abspath $(BIN_DIR))/test_dfn_rate_bridge"
	"$(abspath $(BIN_DIR))/test_mono_dfn"
	"$(abspath $(BIN_DIR))/test_mcab_sample_post"
	"$(abspath $(BIN_DIR))/test_readme_flow"
	"$(abspath $(BIN_DIR))/test_ulcnet_nn_stage"
	"$(abspath $(BIN_DIR))/test_dfn_hop"
	"$(abspath $(BIN_DIR))/mono_alignulcnet"
	"$(abspath $(BIN_DIR))/mono_alignulcnet" --delay-num-filters 1
	"$(abspath $(BIN_DIR))/mono_aec_dfn_res"
	"$(abspath $(BIN_DIR))/mono_aec_dfn_res" --delay-num-filters 1 --cng --atten-lim 12
	"$(abspath $(BIN_DIR))/mono_aec_dfn_res" --sample-rate 48000
	"$(abspath $(BIN_DIR))/mono_aec_dfn_res" --sample-rate 8000
	"$(abspath $(BIN_DIR))/mono_aec_dfn_res" --delay-mode fixed --fixed-delay 64
# Malformed numbers (DFN2 and ULCNet examples) are refused at parse time with the documented exit code 2,
# not merely "some failure" (a crash or a later pool rejection is nonzero too).
	"$(abspath $(BIN_DIR))/mono_aec_dfn_res" --sample-rate 16000Hz >/dev/null 2>&1; test $$? -eq 2
	"$(abspath $(BIN_DIR))/mono_aec_dfn_res" --delay-mode fixed --fixed-delay 12x >/dev/null 2>&1; test $$? -eq 2
	"$(abspath $(BIN_DIR))/mono_aec_dfn_res" --atten-lim 12dB >/dev/null 2>&1; test $$? -eq 2
# An option followed by another option has no value (exit 2); "-x" is a value, here a file that is missing (exit 1).
	for a in "--erb-fwd --cng --erb-inv x" "--erb-inv --cng --erb-fwd x" "--erb-fwd" "--atten-lim --cng" "--sample-rate --cng" "--delay-num-filters --cng" "--delay-mode --" "--fixed-delay --"; do "$(abspath $(BIN_DIR))/mono_aec_dfn_res" $$a >/dev/null 2>&1; test $$? -eq 2 || exit 1; done
	"$(abspath $(BIN_DIR))/mono_aec_dfn_res" --erb-fwd -x --erb-inv -y >/dev/null 2>&1; test $$? -eq 1
	"$(abspath $(BIN_DIR))/mono_alignulcnet" --delay-num-filters 3x >/dev/null 2>&1; test $$? -eq 2
	"$(abspath $(BIN_DIR))/mono_alignulcnet" --delay-mode fixed --fixed-delay 12x >/dev/null 2>&1; test $$? -eq 2
# Empty, leading or trailing garbage and out-of-int-range numbers.
	for v in "" " 3" "3 " x3 +-1 99999999999 999999999999999999999999; do "$(abspath $(BIN_DIR))/mono_alignulcnet" --delay-num-filters "$$v" >/dev/null 2>&1; test $$? -eq 2 || exit 1; done

# example-adapter: build + RUN the REFERENCE ONLY
# board-adapter host simulation (example_board_adapter.c) standalone -- same
# build-then-run shape as `test:` above. Its own exit code is the smoke-test
# signal: nonzero if any positive-flow step or negative-demonstration check
# failed, so a red `make example-adapter` fails the build the same way a red
# `make test` does.
example-adapter: $(BIN_DIR)/example_board_adapter
	"$(abspath $(BIN_DIR))/example_board_adapter"

# Root-name alias for the binary itself (mirrors test_mono_chain: below)
# -- `example-adapter` (above) is the build-AND-run entry point; this is just
# "build the binary" for whoever wants that split.
example_board_adapter: $(BIN_DIR)/example_board_adapter

# libs: alias that builds the three producer archives (an automatic side
# effect of the FORCE rules below) and asserts the local DFN pipeline archive
# exists, for tooling that invokes `make libs`.
libs: $(COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(DFN_LIB_TARGET)

.PHONY: test_target_abi target-abi-test
test_target_abi: $(BIN_DIR)/test_target_abi
target-abi-test: $(BIN_DIR)/test_target_abi
	"$(abspath $(BIN_DIR))/test_target_abi"

$(BIN_DIR)/test_target_abi: $(TARGET_ABI_SRCS) $(COMMON_LIB) $(AEC_LIB) $(NR_LIB) | _cfg_guard
	@mkdir -p "$(OBJ_DIR)/target_abi"
	@rm -f "$(OBJ_DIR)/target_abi/"*.o
	@for source in $(TARGET_ABI_SRCS); do \
	  object="$(OBJ_DIR)/target_abi/$$(basename "$$source" .c).o"; \
	  $(CC) $(CFLAGS) $(ULCNET_GRID_CFLAGS) $(TARGET_ABI_INCLUDES) -MD -MP \
	    -c "$$source" -o "$$object" || exit $$?; \
	done
	$(LINK) -o $@ $(OBJ_DIR)/target_abi/*.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDFLAGS)


# Root-name phony aliases: `make aec_nr_pipeline`, etc. build the artifact at
# $(BIN_DIR)/<name>, not flat in this directory.
aec_nr_pipeline: $(BIN_DIR)/aec_nr_pipeline
aec_nr_pipeline_static: $(BIN_DIR)/aec_nr_pipeline_static
mono_alignulcnet: $(BIN_DIR)/mono_alignulcnet
mono_aec_dfn_res: $(BIN_DIR)/mono_aec_dfn_res
test_mono_chain: $(BIN_DIR)/test_mono_chain
test_mono_ulcnet: $(BIN_DIR)/test_mono_ulcnet
test_ulcnet_accelerator_adapter: $(BIN_DIR)/test_ulcnet_accelerator_adapter
test_dfn_res_stage: $(BIN_DIR)/test_dfn_res_stage
test_dfn_rate_bridge: $(BIN_DIR)/test_dfn_rate_bridge
test_mono_dfn: $(BIN_DIR)/test_mono_dfn
test_mcab_sample_post: $(BIN_DIR)/test_mcab_sample_post
test_readme_flow: $(BIN_DIR)/test_readme_flow
test_dfn_hop: $(BIN_DIR)/test_dfn_hop
test_ulcnet_nn_stage: $(BIN_DIR)/test_ulcnet_nn_stage

FORCE:

# Each producer archive is a FORCE-recipe target: it always re-invokes the producer's own build (a
# no-op when up to date) and only touches the archive's mtime if something was rebuilt, so the
# binaries below relink only then. COMMON_DIR/COMMON_LIB are forwarded to the lib/aec and lib/nr
# sub-makes even though `lib` never links audiolib, so their CFG_SIG (hence output path) matches
# what the dispatch query resolved.
$(COMMON_LIB): FORCE
	+$(MAKE) -C $(COMMON_DIR) $(FWD) lib
	@test -f '$@' || { echo "ERROR: audiolib did not produce $@ (query/build divergence -- supported forwarded controls are BACKEND/SIMD/WERROR/CC/CXX/EXTRA_CFLAGS/NO_STDIO/TOOLCHAIN_CHECK)"; exit 1; }

$(AEC_LIB): FORCE
	+$(MAKE) -C $(AEC_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB=$(COMMON_LIB) lib
	@test -f '$@' || { echo "ERROR: lib/aec did not produce $@ (query/build divergence)"; exit 1; }

$(NR_LIB): FORCE
	+$(MAKE) -C $(NR_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB=$(COMMON_LIB) lib
	@test -f '$@' || { echo "ERROR: lib/nr did not produce $@ (query/build divergence)"; exit 1; }

# Compile with $(CC) (always a plain-C driver, -std=gnu99 valid) then link
# with $(LINK) (c++ -- the NE10 archive's one C++ TU needs it). Link order
# matters (see the LDFLAGS comment above): program objects,
# libaec.a, libanr.a, libaudio_common.a, -lm.
$(BIN_DIR)/aec_nr_pipeline: $(OBJ_DIR)/aec_nr_pipeline.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/aec_nr_pipeline.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDFLAGS)
	@echo "aec_nr_pipeline [$(BACKEND)] -> $@"

$(BIN_DIR)/aec_nr_pipeline_static: $(OBJ_DIR)/aec_nr_pipeline_static.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/aec_nr_pipeline_static.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDFLAGS)
	@echo "aec_nr_pipeline_static [$(BACKEND)] -> $@"

$(BIN_DIR)/test_mono_chain: $(OBJ_DIR)/test_mono_chain.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_mono_chain.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDFLAGS)
	@echo "test_mono_chain [$(BACKEND)] -> $@"

# The Align-ULCNet programs link the stage and the AIAEC pre/post objects
# directly. They are applications, not additional libraries layered over the component libs.
ULCNET_PIPELINE_OBJS = $(OBJ_DIR)/ulcnet_nn_stage.o $(OBJ_DIR)/ulcnet_process.o

# The stage only READS the descriptor a runtime published, but it validates it through
# ulcnet_model_io, which the test also uses to BUILD descriptors for the far-input contract gate.
$(BIN_DIR)/test_mono_ulcnet: $(OBJ_DIR)/test_mono_ulcnet.o $(ULCNET_PIPELINE_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(AEC_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_mono_ulcnet.o $(ULCNET_PIPELINE_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(AEC_LIB) $(COMMON_LIB) $(LDFLAGS)
	@echo "test_mono_ulcnet [$(BACKEND)] -> $@"

# The adapter drives the pre/post class, so anything linking the adapter
# links ulcnet_prepost.o with it. The class in turn references
# ulcnet_process.c's analysis/synthesis entry points unconditionally -- the
# ULCNET_IO_FREQ instances the adapter builds never CALL them, but the symbols
# still have to resolve -- which is why the adapter test links ulcnet_process.o
# and, through it, audiolib's FFT wrapper.
ULCNET_ADAPTER_OBJS = $(OBJ_DIR)/ulcnet_accelerator_adapter.o \
                      $(OBJ_DIR)/ulcnet_prepost.o \
                      $(OBJ_DIR)/ulcnet_model_io.o

$(BIN_DIR)/mono_alignulcnet: $(OBJ_DIR)/mono_alignulcnet_main.o $(ULCNET_ADAPTER_OBJS) $(ULCNET_PIPELINE_OBJS) $(AEC_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/mono_alignulcnet_main.o $(ULCNET_ADAPTER_OBJS) $(ULCNET_PIPELINE_OBJS) $(AEC_LIB) $(COMMON_LIB) $(LDFLAGS)
	@echo "mono_alignulcnet [$(BACKEND)] -> $@"

$(BIN_DIR)/test_ulcnet_accelerator_adapter: $(OBJ_DIR)/test_ulcnet_accelerator_adapter.o $(ULCNET_ADAPTER_OBJS) $(OBJ_DIR)/ulcnet_process.o $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_ulcnet_accelerator_adapter.o $(ULCNET_ADAPTER_OBJS) $(OBJ_DIR)/ulcnet_process.o $(COMMON_LIB) $(LDFLAGS)

$(BIN_DIR)/test_ulcnet_nn_stage: $(OBJ_DIR)/test_ulcnet_nn_stage.o $(OBJ_DIR)/ulcnet_nn_stage.o $(OBJ_DIR)/ulcnet_process.o $(OBJ_DIR)/ulcnet_model_io.o $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_ulcnet_nn_stage.o $(OBJ_DIR)/ulcnet_nn_stage.o $(OBJ_DIR)/ulcnet_process.o $(OBJ_DIR)/ulcnet_model_io.o $(COMMON_LIB) $(LDFLAGS)

# The mono DFN2 program composes lib/aec's Aec (its RES gain) with the DFN archive; it has no
# denoiser.
$(BIN_DIR)/mono_aec_dfn_res: $(OBJ_DIR)/mono_aec_dfn_res_main.o $(DFN_LIB_TARGET) $(AEC_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/mono_aec_dfn_res_main.o $(DFN_LIB_TARGET) $(AEC_LIB) $(COMMON_LIB) $(LDFLAGS)
	@echo "mono_aec_dfn_res [$(BACKEND)] -> $@"

$(BIN_DIR)/test_mcab_sample_post: $(OBJ_DIR)/test_mcab_sample_post.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_mcab_sample_post.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDFLAGS)
	@echo "test_mcab_sample_post [$(BACKEND)] -> $@"

# test_readme_flow: the call-flow examples of README_01_latency_call_flow.md, compiled and run (chain a stage, DFN and ULCNet phased hops).
$(BIN_DIR)/test_readme_flow: $(OBJ_DIR)/test_readme_flow.o $(ULCNET_PIPELINE_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(DFN_LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_readme_flow.o $(ULCNET_PIPELINE_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(DFN_LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDFLAGS)
	@echo "test_readme_flow [$(BACKEND)] -> $@"

$(BIN_DIR)/test_dfn_res_stage: $(OBJ_DIR)/test_dfn_res_stage.o $(DFN_STAGE_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_dfn_res_stage.o $(DFN_STAGE_OBJS) $(COMMON_LIB) $(LDFLAGS)
	@echo "test_dfn_res_stage [$(BACKEND)] -> $@"

$(BIN_DIR)/test_dfn_rate_bridge: $(OBJ_DIR)/test_dfn_rate_bridge.o $(OBJ_DIR)/dfn_rate_bridge.o $(DFN_STAGE_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_dfn_rate_bridge.o $(OBJ_DIR)/dfn_rate_bridge.o $(DFN_STAGE_OBJS) $(COMMON_LIB) $(LDFLAGS)
	@echo "test_dfn_rate_bridge [$(BACKEND)] -> $@"

$(BIN_DIR)/test_dfn_hop: $(OBJ_DIR)/test_dfn_hop.o $(DFN_PIPELINE_OBJS) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_dfn_hop.o $(DFN_PIPELINE_OBJS) $(COMMON_LIB) $(LDFLAGS)

$(BIN_DIR)/test_mono_dfn: $(OBJ_DIR)/test_mono_dfn.o $(DFN_LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_mono_dfn.o $(DFN_LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDFLAGS)
	@echo "test_mono_dfn [$(BACKEND)] -> $@"

$(BIN_DIR)/example_board_adapter: $(OBJ_DIR)/example_board_adapter.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/example_board_adapter.o $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDFLAGS)
	@echo "example_board_adapter [$(BACKEND)] -> $@"
endif

# --- Query targets -----------------------------------------------------
# Report this invocation's own build identity (the producer SIGs are empty unless COMMON_LIB/AEC_LIB/
# NR_LIB were passed). Nothing above ever $(info)s or $(warning)s unconditionally, so stdout is
# exactly the one recipe-time @echo line, safe to capture with `$(MAKE) -s ... print-lib-path` as the
# dispatch recipe and the audit/publish recipes do.
.PHONY: print-bin-dir print-obj-dir print-lib-path
# print-bin-dir / print-lib-path / print-obj-dir dispatch through the same three-producer phase-1
# resolution as the build goals, so a bare `make -s print-<x>` prints the location this configuration
# actually uses. Passing COMMON_LIB/AEC_LIB/NR_LIB explicitly skips the queries.
ifeq ($(COMMON_LIB),)
print-bin-dir print-lib-path print-obj-dir:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audiolib archive path (COMMON_DIR=$(COMMON_DIR))" >&2; exit 1; }; \
	 aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 nr="$$($(MAKE) -s --no-print-directory -C $(NR_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 $(MAKE) -s --no-print-directory $@ COMMON_LIB="$$ac" AEC_LIB="$$aec" NR_LIB="$$nr"
else
print-bin-dir:
	@echo $(abspath $(BIN_DIR))
print-lib-path:
	@echo $(abspath $(DFN_LIB_TARGET))
print-obj-dir:
	@echo $(abspath $(OBJ_DIR))
endif

# test-fp-contract: disassemble the pipeline objects that run windowed
# overlap-add and assert they emitted no fma-class instruction. Part of the
# release gate: -ffp-contract=off is a compile flag, fma-free is a property of
# the generated code, and only this says which one actually happened. The
# script builds each object itself and queries print-obj-dir from the same
# configuration, so it can never audit a stale signature's leftovers.
.PHONY: test-fp-contract
test-fp-contract:
	@scripts/audit_fp_contract.sh $(BACKEND)

# audit-no-stdio: builds libaudio_pipeline_dfn.a with NO_STDIO=1 and asserts, with `nm` over the delivered
# keyed archive itself, that none of fprintf/printf/puts/fputs/stderr/__stderrp is in it, and that its
# resolved path differs from the default (NO_STDIO=0) config's path so the two cannot alias. `ar -t`
# confirms the members (dfn_rate_bridge.o, dfn_res_stage.o, dfn2_prepost.o, dfn2_model_io.o,
# dfn2_process.o). The regex tolerates a
# leading underscore and matches the last field of nm's NAME column, as
# ../../audiolib/common/scripts/audit_alloc_symbols.sh does, so it works with macOS/BSD and GNU nm.
# Prints PASS/FAIL; non-zero exit on FAIL.
.PHONY: audit-no-stdio
audit-no-stdio:
	@echo "audit-no-stdio: building libaudio_pipeline_dfn.a (BACKEND=$(BACKEND), NO_STDIO=1)"
	@$(MAKE) --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 libaudio_pipeline_dfn.a >/dev/null
	@lib_nostdio="$$($(MAKE) -s --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 print-lib-path)"; \
	 lib_default="$$($(MAKE) -s --no-print-directory BACKEND=$(BACKEND) NO_STDIO=0 print-lib-path)"; \
	 if [ "$$lib_nostdio" = "$$lib_default" ]; then \
	   echo "FATAL: NO_STDIO=1 archive path COLLIDES with the default (NO_STDIO=0) archive path ($$lib_nostdio)" >&2; \
	   exit 1; \
	 fi; \
	 if [ ! -f "$$lib_nostdio" ]; then \
	   echo "FAIL: $$lib_nostdio not found after building libaudio_pipeline_dfn.a with NO_STDIO=1"; \
	   exit 1; \
	 fi; \
	 for m in dfn_rate_bridge dfn_res_stage dfn2_prepost dfn2_model_io dfn2_process; do \
	   ar -t "$$lib_nostdio" | grep -q "^$$m\.o$$" || { echo "FAIL: $$lib_nostdio missing $$m.o member"; exit 1; }; \
	 done; \
	 echo "audit-no-stdio: nm -A $$lib_nostdio | grep -E '^_?(fprintf|printf|puts|fputs|stderr|__stderrp)$$' (NAME column)"; \
	 hits=$$(nm -A "$$lib_nostdio" | awk '{ name = $$NF; if (name ~ /^_?(fprintf|printf|puts|fputs|stderr|__stderrp)$$/) print }'); \
	 if [ -n "$$hits" ]; then \
	   echo "FAIL: forbidden stdio symbol(s) found in $$lib_nostdio (NO_STDIO=1 build):"; \
	   echo "$$hits"; \
	   exit 1; \
	 fi; \
	 echo "PASS: no fprintf/printf/puts/fputs/stderr/__stderrp symbols in $$lib_nostdio (NO_STDIO=1); path differs from default-config path $$lib_default"

# audit-no-stdio-stack (full-image gate): builds the delivered archives with NO_STDIO=1 (the
# dispatch forwards it to audiolib / lib/aec / lib/nr; lib/aec compiles its trace/debug TU out under
# it), nm-gates each archive, then links test_no_stdio_stack.c (a minimal main that runs
# query->init->process->reset->destroy over Aec, the denoiser and the synthesis, using no stdio) and nm-gates the
# executable, so a stdio symbol from any archive fails at the image level. The stdio set is
# fprintf/vfprintf/printf/puts/fputs/stderr/__stderrp/fputc/fwrite.
.PHONY: audit-no-stdio-stack
audit-no-stdio-stack:
	@echo "audit-no-stdio-stack: building the full stack (BACKEND=$(BACKEND), NO_STDIO=1)"
	@$(MAKE) --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 libaudio_pipeline_dfn.a >/dev/null
	@bd="$$($(MAKE) -s --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 print-bin-dir)"; \
	 dfnpl="$$bd/libaudio_pipeline_dfn.a"; \
	 ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 print-lib-path)"; \
	 aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 nr="$$($(MAKE) -s --no-print-directory -C $(NR_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 $(MAKE) --no-print-directory -C $(COMMON_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 lib >/dev/null; \
	 $(MAKE) --no-print-directory -C $(AEC_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" lib >/dev/null; \
	 $(MAKE) --no-print-directory -C $(NR_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" lib >/dev/null; \
	 pat='^_?(fprintf|vfprintf|printf|puts|fputs|stderr|__stderrp|fputc|fwrite)$$'; \
	 for lib in "$$dfnpl" "$$aec" "$$nr" "$$ac"; do \
	   test -f "$$lib" || { echo "FAIL: archive not found: $$lib"; exit 1; }; \
	   hits=$$(nm -A "$$lib" | awk -v re="$$pat" '{ n=$$NF; if (n ~ re) print }'); \
	   if [ -n "$$hits" ]; then echo "FAIL: stdio symbol(s) in delivered archive $$lib:"; echo "$$hits"; exit 1; fi; \
	   echo "  archive clean: $$lib"; \
	 done; \
	 mkdir -p "$$bd"; \
	 $(CC) $(CFLAGS) -c -o "$$bd/test_no_stdio_stack.o" mono_aec_nr_res/tests/test_no_stdio_stack.c; \
	 $(LINK) -o "$$bd/test_no_stdio_stack" "$$bd/test_no_stdio_stack.o" "$$aec" "$$nr" "$$ac" -lm; \
	 hits=$$(nm "$$bd/test_no_stdio_stack" | awk -v re="$$pat" '{ n=$$NF; if (n ~ re) print }'); \
	 if [ -n "$$hits" ]; then echo "FAIL: stdio symbol(s) in the linked EXECUTABLE:"; echo "$$hits"; exit 1; fi; \
	 "$$bd/test_no_stdio_stack" || { rc=$$?; echo "FAIL: test_no_stdio_stack exit $$rc"; exit 1; }; \
	 echo "PASS: full NO_STDIO=1 stack -- 4 delivered archives scanned + conventional-stack executable clean ($$pat tier), runtime exit 0"

# publish: immutable, content-addressed release under $(DIST_ROOT), same design as audiolib's publish
# (see its Makefile): dist/<backend>/<cfg_sig>/ plus an atomic dist/<backend>/current symlink, a
# per-backend mkdir-lock serialising concurrent publishes, and MANIFEST.txt with relative paths,
# per-file sha256, the three resolved producer identities and the git commit of each of the four
# repos (audiolib, lib/aec, lib/nr via `git -C`, and this repo). PUBLISH_ARTIFACTS needs
# $(COMMON_LIB)/$(AEC_LIB)/$(NR_LIB), so `publish` goes through the same two-phase dispatch as
# all/test/libs.

# - ATTEST is one event, one file: exactly one new attest-<utc>-<commit>[-dirty]-<seq>.txt per publish
#   event (idempotent republishes included), installed with the atomic_symlink_swap helper's
#   `--excl-install` mode (write-temp + link(2), an atomic no-clobber). A same-second name collision
#   regenerates the event id with the next <seq> (capped at 999); an existing attestation is never
#   overwritten.
# - Every commit recorded is the full 40-hex OID, for this repo and the three producers. ATTEST/ names
#   the exact checkout of the whole stack, while MANIFEST stays deterministic (config, tool
#   identities, per-file SHA-256), so the same release id always has the same MANIFEST bytes.

# - Per repo, an unknown identity is checked first and refused unconditionally (no commit OID, no
#   provenance). Only when all four repos have a commit are TRACKED-dirty (`git status --porcelain
#   -uno`) and UNTRACKED (`git ls-files --others --exclude-standard`; this repo's list has the
#   AINR/GTCRN/ and AINR/gtcrn_github/ prefixes whitelisted) computed. Gate-then-hash: the dirty gate
#   runs for all four repos, then the untracked gate, and only then is anything hashed. The escape
#   hatches ALLOW_DIRTY_PUBLISH / ALLOW_UNTRACKED_PUBLISH stay separate and record dirty_diff_sha256 /
#   untracked_tree_sha256 (via hash_untracked(), see _publish_impl).

# - Attestations are unsigned: traceability, not authenticity. MANIFEST byte-verification is the
#   integrity check; ATTEST is the provenance log.

# `publish` is the locked driver (a MAKEFLAGS word-scan makes -n/-q/-t write nothing; see its own
# comment); _publish_impl is internal (direct invocation skips the lock) and uses the same
# three-producer dispatch as `all`/`test`.
# =============================================================================
PUBLISH_ARTIFACTS = $(DFN_LIB_TARGET) $(BIN_DIR)/aec_nr_pipeline $(BIN_DIR)/aec_nr_pipeline_static $(BIN_DIR)/mono_alignulcnet $(BIN_DIR)/mono_aec_dfn_res

# Release gate (deliberately NOT wired into `test:` -- it runs a full real
# publish): publishes into a throwaway DIST_ROOT and asserts every
# PUBLISH_ARTIFACTS basename appears in the resulting MANIFEST.txt. Guards
# the "new artifact built but never shipped" failure mode: adding a target
# without extending PUBLISH_ARTIFACTS makes this gate fail instead of
# silently publishing an incomplete set. The ULCNet application binary is
# published directly; there is no separate application wrapper archive.
.PHONY: test-publish-manifest
test-publish-manifest:
	@tmp=$$(mktemp -d) || exit 1; \
	 $(MAKE) --no-print-directory publish DIST_ROOT=$$tmp ALLOW_DIRTY_PUBLISH=1 >/dev/null 2>&1 || \
	   { echo "FAIL: publish into throwaway DIST_ROOT failed"; rm -rf "$$tmp"; exit 1; }; \
	 manifest=$$(find "$$tmp" -name MANIFEST.txt | head -1); \
	 test -n "$$manifest" || { echo "FAIL: no MANIFEST.txt produced"; rm -rf "$$tmp"; exit 1; }; \
	 rc=0; \
	 for a in $(notdir $(PUBLISH_ARTIFACTS)); do \
	   grep -q "$$a" "$$manifest" || { echo "FAIL: $$a missing from MANIFEST.txt"; rc=1; }; \
	 done; \
	 [ $$rc -eq 0 ] && echo "publish manifest OK ($(words $(PUBLISH_ARTIFACTS)) artifacts verified)"; \
	 rm -rf "$$tmp"; exit $$rc
DIST_BACKEND_DIR  = $(DIST_ROOT)/$(BACKEND)
DIST_LOCK         = $(DIST_ROOT)/.lock.$(BACKEND)

.PHONY: publish
# Dry-run guard: this recipe mentions $(MAKE), so GNU make runs it for real under -n/-q/-t, which would
# create $(DIST_ROOT) and take the publish lock on every dry run. A MAKEFLAGS word-scan branches:
#   -t  print a note and exit 0 without recursing (recursing would let touch semantics bump the mtimes
#       of the artifact prerequisites; publish is a phony action, not a build product). Checked first:
#       for `make -nt publish`, recursing for -n would hand the child both flags and make would touch.
#   -q  exit 1 with no output (the documented "needs updating" answer; publish is phony).
#   -n  recurse; the child inherits -n and only prints the pin check and _publish_impl commands.

# All three write nothing. The scan handles bundled short flags as the first word ("n", "sn") or as
# separate dashed words, long options ("--"), and a literal "--" before command-line variables (which
# can contain n/q/t, e.g. BACKEND=ne10), hence the explicit skips. Normal invocations lock first.
publish:
	+@has_n=0; has_q=0; has_t=0; \
	for w in $(MAKEFLAGS); do \
	  case "$$w" in --) break ;; --*|*=*) continue ;; -n|-q|-t) ;; -*) continue ;; esac; \
	  case "$$w" in *n*) has_n=1 ;; esac; \
	  case "$$w" in *q*) has_q=1 ;; esac; \
	  case "$$w" in *t*) has_t=1 ;; esac; \
	done; \
	if [ "$$has_t" = 1 ]; then echo "publish: -t (touch) is a no-op here -- publish is an action, not a build product; nothing was touched"; exit 0; fi; \
	if [ "$$has_q" = 1 ]; then exit 1; fi; \
	if [ "$$has_n" = 1 ]; then exec $(MAKE) --no-print-directory _publish_checked; fi; \
	mkdir -p $(DIST_ROOT); \
	if ! mkdir "$(DIST_LOCK)" 2>/dev/null; then \
	  echo "FATAL: publish lock $(DIST_LOCK) already held (concurrent 'make publish BACKEND=$(BACKEND)'?)" >&2; \
	  exit 1; \
	fi; \
	trap 'rmdir "$(DIST_LOCK)" 2>/dev/null' EXIT INT TERM; \
	$(MAKE) --no-print-directory _publish_checked

# Keep dependency verification inside the publish lock and ahead of artifact
# prerequisites. A normal prerequisite on _publish_impl would be insufficient
# under -j because Make may build sibling prerequisites concurrently; a single
# shell line joined with && keeps the ordering even under -i / .IGNORE.
.PHONY: _publish_checked
_publish_checked:
	+$(MAKE) --no-print-directory _check_pins_impl && $(MAKE) --no-print-directory _publish_impl

# _publish_impl records the provenance of four repos (this repo and the three producers), each as a
# full 40-hex commit OID plus a dirty flag, so ATTEST/ names the source state of the whole stack.

# hash_untracked() (defined at the top of the recipe, the same helper as audiolib's): $1 is a path
# prefix; it reads an untracked-path list on stdin, sorts it (`LC_ALL=C sort`) and emits one
# fixed-field record per path: "L <sha256(path)> <sha256(readlink)>" for a symlink, "F <mode>
# <sha256(path)> <sha256(content)>" for a regular file, "X <path>" for anything else or any
# stat/readlink/shasum failure (fail-closed, never an empty hash). Variable-length values are hashed
# first so records concatenate unambiguously. It runs once per repo ("./" over the whitelist-filtered
# list; "$(COMMON_DIR)/", "$(AEC_DIR)/", "$(NR_DIR)/" over the raw lists) into its own record file,
# and each untracked_tree_sha256 is `shasum -a 256` of that file.

# Per repo, an unknown identity is checked first and is an unconditional FATAL. Each producer dir is
# guarded with `[ -z "$(X_DIR)" ] || [ ! -d "$(X_DIR)" ]` before any `git -C`, so an empty or missing
# dir fails naming the producer. Once all four have a commit, the dirty gate (all four) runs, then the
# untracked gate (all four), each a FATAL naming the repo unless its escape hatch is set; only then
# are the dirty_diff_sha256 hashes (repos with dirty=1) and the record files (repos with untracked=1)
# computed.

# ALLOW_DIRTY_PUBLISH=1 / ALLOW_UNTRACKED_PUBLISH=1 record allow_dirty_publish=1 /
# allow_untracked_publish=1 plus the per-repo dirty_diff_sha256 / untracked_tree_sha256. An X record is
# always a FATAL naming the repo: a publish must not proceed as though it could account for a source
# it cannot hash.
ifneq ($(COMMON_LIB),)
.PHONY: _publish_impl
_publish_impl: $(PUBLISH_ARTIFACTS)
	@set -e; \
	work="$$(mktemp -d)"; \
	tmp="$(DIST_BACKEND_DIR)/.stage.$$$$"; \
	trap 'rm -rf "$$tmp" "$$work"' EXIT INT TERM; \
	hash_untracked() { \
	  LC_ALL=C sort | while IFS= read -r p; do \
	    fp="$$1$$p"; \
	    ph="$$(printf '%s' "$$p" | shasum -a 256 | cut -d' ' -f1)"; \
	    [ "$${#ph}" -eq 64 ] || { printf 'X %s\n' "$$p"; continue; }; \
	    if [ -L "$$fp" ]; then \
	      tgt="$$(readlink "$$fp")" || { printf 'X %s\n' "$$p"; continue; }; \
	      th="$$(printf '%s' "$$tgt" | shasum -a 256 | cut -d' ' -f1)"; \
	      [ "$${#th}" -eq 64 ] || { printf 'X %s\n' "$$p"; continue; }; \
	      printf 'L %s %s\n' "$$ph" "$$th"; \
	    elif [ -f "$$fp" ]; then \
	      m="$$(stat -f '%Lp' "$$fp" 2>/dev/null || stat -c '%a' "$$fp" 2>/dev/null)"; \
	      ch="$$(shasum -a 256 "$$fp" 2>/dev/null | cut -d' ' -f1)"; \
	      { [ -n "$$m" ] && [ "$${#ch}" -eq 64 ]; } || { printf 'X %s\n' "$$p"; continue; }; \
	      printf 'F %s %s %s\n' "$$m" "$$ph" "$$ch"; \
	    else printf 'X %s\n' "$$p"; fi; \
	  done; \
	}; \
	commit="$$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- pipelines (this repo) has no git identity here (not a git checkout)." >&2; \
	  echo "  Without an identity the attestation cannot name the source state OR enumerate untracked" >&2; \
	  echo "  content, so NO escape hatch admits this. Publish from a real checkout or a scratch clone" >&2; \
	  echo "  with a disposable commit instead." >&2; \
	  exit 1; \
	fi; \
	dirty=0; [ -n "$$(git status --porcelain -uno 2>/dev/null)" ] && dirty=1; \
	unt_list="$$(git ls-files --others --exclude-standard 2>/dev/null | grep -vE '^AINR/(GTCRN|gtcrn_github)/' || true)"; \
	untracked=0; [ -n "$$unt_list" ] && untracked=1; \
	if [ -z "$(COMMON_DIR)" ] || [ ! -d "$(COMMON_DIR)" ]; then \
	  echo "FATAL: publish refused -- audiolib has no git identity (not a git checkout, or COMMON_DIR=$(COMMON_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	ac_commit="$$(git -C "$(COMMON_DIR)" rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$ac_commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- audiolib has no git identity (not a git checkout, or COMMON_DIR=$(COMMON_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	ac_dirty=0; [ -n "$$(git -C "$(COMMON_DIR)" status --porcelain -uno 2>/dev/null)" ] && ac_dirty=1; \
	ac_unt_list="$$(git -C "$(COMMON_DIR)" ls-files --others --exclude-standard 2>/dev/null)"; \
	ac_untracked=0; [ -n "$$ac_unt_list" ] && ac_untracked=1; \
	if [ -z "$(AEC_DIR)" ] || [ ! -d "$(AEC_DIR)" ]; then \
	  echo "FATAL: publish refused -- lib/aec has no git identity (not a git checkout, or AEC_DIR=$(AEC_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	aec_commit="$$(git -C "$(AEC_DIR)" rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$aec_commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- lib/aec has no git identity (not a git checkout, or AEC_DIR=$(AEC_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	aec_dirty=0; [ -n "$$(git -C "$(AEC_DIR)" status --porcelain -uno 2>/dev/null)" ] && aec_dirty=1; \
	aec_unt_list="$$(git -C "$(AEC_DIR)" ls-files --others --exclude-standard 2>/dev/null)"; \
	aec_untracked=0; [ -n "$$aec_unt_list" ] && aec_untracked=1; \
	if [ -z "$(NR_DIR)" ] || [ ! -d "$(NR_DIR)" ]; then \
	  echo "FATAL: publish refused -- lib/nr has no git identity (not a git checkout, or NR_DIR=$(NR_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	nr_commit="$$(git -C "$(NR_DIR)" rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$nr_commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- lib/nr has no git identity (not a git checkout, or NR_DIR=$(NR_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	nr_dirty=0; [ -n "$$(git -C "$(NR_DIR)" status --porcelain -uno 2>/dev/null)" ] && nr_dirty=1; \
	nr_unt_list="$$(git -C "$(NR_DIR)" ls-files --others --exclude-standard 2>/dev/null)"; \
	nr_untracked=0; [ -n "$$nr_unt_list" ] && nr_untracked=1; \
	adp=0; [ "$(ALLOW_DIRTY_PUBLISH)" = "1" ] && adp=1; \
	aup=0; [ "$(ALLOW_UNTRACKED_PUBLISH)" = "1" ] && aup=1; \
	if [ "$$adp" != "1" ]; then \
	  bad=0; \
	  if [ "$$dirty" = "1" ]; then echo "FATAL: publish refused -- pipelines (this repo) working tree has uncommitted TRACKED changes." >&2; bad=1; fi; \
	  if [ "$$ac_dirty" = "1" ]; then echo "FATAL: publish refused -- audiolib working tree has uncommitted TRACKED changes." >&2; bad=1; fi; \
	  if [ "$$aec_dirty" = "1" ]; then echo "FATAL: publish refused -- lib/aec working tree has uncommitted TRACKED changes." >&2; bad=1; fi; \
	  if [ "$$nr_dirty" = "1" ]; then echo "FATAL: publish refused -- lib/nr working tree has uncommitted TRACKED changes." >&2; bad=1; fi; \
	  if [ "$$bad" = "1" ]; then \
	    echo "  Commit (or clone a clean checkout) for a board deliverable, or pass ALLOW_DIRTY_PUBLISH=1" >&2; \
	    echo "  for a dev publish -- the escape hatch is recorded in the attestation (covers TRACKED changes only)." >&2; \
	    exit 1; \
	  fi; \
	fi; \
	if [ "$$aup" != "1" ]; then \
	  bad=0; \
	  if [ "$$untracked" = "1" ]; then \
	    echo "FATAL: publish refused -- pipelines (this repo) contains UNTRACKED files outside the AINR/GTCRN|gtcrn_github whitelist:" >&2; \
	    printf '%s\n' "$$unt_list" | head -5 | sed 's/^/    /' >&2; \
	    bad=1; \
	  fi; \
	  if [ "$$ac_untracked" = "1" ]; then \
	    echo "FATAL: publish refused -- audiolib contains UNTRACKED files:" >&2; \
	    printf '%s\n' "$$ac_unt_list" | head -5 | sed 's/^/    /' >&2; \
	    bad=1; \
	  fi; \
	  if [ "$$aec_untracked" = "1" ]; then \
	    echo "FATAL: publish refused -- lib/aec contains UNTRACKED files:" >&2; \
	    printf '%s\n' "$$aec_unt_list" | head -5 | sed 's/^/    /' >&2; \
	    bad=1; \
	  fi; \
	  if [ "$$nr_untracked" = "1" ]; then \
	    echo "FATAL: publish refused -- lib/nr contains UNTRACKED files:" >&2; \
	    printf '%s\n' "$$nr_unt_list" | head -5 | sed 's/^/    /' >&2; \
	    bad=1; \
	  fi; \
	  if [ "$$bad" = "1" ]; then \
	    echo "  Track/remove them for a board deliverable, or pass ALLOW_UNTRACKED_PUBLISH=1 -- the escape" >&2; \
	    echo "  hatch is recorded in the attestation with untracked_tree_sha256 over their exact contents." >&2; \
	    exit 1; \
	  fi; \
	fi; \
	ddiff=""; [ "$$dirty" = "1" ] && ddiff="$$(git diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	ac_ddiff=""; [ "$$ac_dirty" = "1" ] && ac_ddiff="$$(git -C "$(COMMON_DIR)" diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	aec_ddiff=""; [ "$$aec_dirty" = "1" ] && aec_ddiff="$$(git -C "$(AEC_DIR)" diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	nr_ddiff=""; [ "$$nr_dirty" = "1" ] && nr_ddiff="$$(git -C "$(NR_DIR)" diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	uhash=""; \
	if [ "$$untracked" = "1" ]; then \
	  printf '%s\n' "$$unt_list" | hash_untracked "./" > "$$work/untracked.records"; \
	  if grep -q '^X ' "$$work/untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path in pipelines (this repo) cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  uhash="$$(shasum -a 256 < "$$work/untracked.records" | cut -d' ' -f1)"; \
	fi; \
	ac_uhash=""; \
	if [ "$$ac_untracked" = "1" ]; then \
	  printf '%s\n' "$$ac_unt_list" | hash_untracked "$(COMMON_DIR)/" > "$$work/ac_untracked.records"; \
	  if grep -q '^X ' "$$work/ac_untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path in audiolib cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/ac_untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  ac_uhash="$$(shasum -a 256 < "$$work/ac_untracked.records" | cut -d' ' -f1)"; \
	fi; \
	aec_uhash=""; \
	if [ "$$aec_untracked" = "1" ]; then \
	  printf '%s\n' "$$aec_unt_list" | hash_untracked "$(AEC_DIR)/" > "$$work/aec_untracked.records"; \
	  if grep -q '^X ' "$$work/aec_untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path in lib/aec cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/aec_untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  aec_uhash="$$(shasum -a 256 < "$$work/aec_untracked.records" | cut -d' ' -f1)"; \
	fi; \
	nr_uhash=""; \
	if [ "$$nr_untracked" = "1" ]; then \
	  printf '%s\n' "$$nr_unt_list" | hash_untracked "$(NR_DIR)/" > "$$work/nr_untracked.records"; \
	  if grep -q '^X ' "$$work/nr_untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path in lib/nr cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/nr_untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  nr_uhash="$$(shasum -a 256 < "$$work/nr_untracked.records" | cut -d' ' -f1)"; \
	fi; \
	dateutc="$$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	stamp='$(ATTEST_STAMP)'; [ -n "$$stamp" ] || stamp="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	$(HOSTCC) -O2 -o "$$work/swapln" $(COMMON_DIR)/tools/atomic_symlink_swap.c || { echo "FATAL: could not build the atomic symlink-swap helper with HOSTCC=$(HOSTCC) (pass HOSTCC=<host compiler>)" >&2; exit 1; }; \
	mkdir -p "$(DIST_BACKEND_DIR)"; \
	rm -rf "$$tmp"; mkdir -p "$$tmp"; \
	cp $(PUBLISH_ARTIFACTS) "$$tmp/"; \
	shas="$$( cd "$$tmp" && for f in *; do shasum -a 256 "$$f"; done )"; \
	content12="$$(printf '%s\n' "$$shas" | sort | shasum -a 256 | cut -c1-12)"; \
	rel_id="$(CFG_SIG)-$$content12"; \
	rel_dir="$(DIST_BACKEND_DIR)/$$rel_id"; \
	{ echo "repo=pipelines"; \
	  echo "release_id=$$rel_id"; \
	  echo "backend=$(BACKEND)"; \
	  echo "cfg_sig=$(CFG_SIG)"; \
	  echo "ac_producer_cfg_sig=$(COMMON_PRODUCER_SIG)"; \
	  echo "aec_producer_cfg_sig=$(AEC_PRODUCER_SIG)"; \
	  echo "nr_producer_cfg_sig=$(NR_PRODUCER_SIG)"; \
	  echo "cc=$(CC)"; echo "cxx=$(CXX)"; echo "ar=$(AR)"; echo "ranlib=$(RANLIB)"; echo "link=$(LINK)"; \
	  echo "cflags=$(CFLAGS)"; \
	  echo "extra_cflags=$(EXTRA_CFLAGS)"; \
	  echo "werror=$(WERROR)"; echo "no_stdio=$(NO_STDIO)"; echo "debug=$(DEBUG)"; \
	  echo "$$shas"; \
	} > "$$tmp/MANIFEST.txt"; \
	suffix=""; [ "$$dirty" = "1" ] && suffix="-dirty"; \
	if [ -d "$$rel_dir" ]; then \
	  for f in "$$tmp"/*; do \
	    bn="$$(basename "$$f")"; \
	    cmp -s "$$f" "$$rel_dir/$$bn" || { echo "FATAL: existing release $$rel_id differs from staged content for $$bn (content-address collision, or the stored release/MANIFEST was modified on disk)" >&2; exit 1; }; \
	  done; \
	  rm -rf "$$tmp"; \
	  echo "pipelines: $$rel_id already published (byte-verified, incl. MANIFEST) -- appending attestation + repointing current"; \
	else \
	  mkdir "$$tmp/ATTEST"; \
	  mv "$$tmp" "$$rel_dir"; \
	fi; \
	mkdir -p "$$rel_dir/ATTEST"; \
	n=1; \
	while :; do \
	  attest_name="attest-$$stamp-$$commit$$suffix-$$(printf '%03d' "$$n").txt"; \
	  { echo "event_id=$${attest_name%.txt}"; \
	    echo "repo=pipelines"; \
	    echo "release_id=$$rel_id"; \
	    echo "cfg_sig=$(CFG_SIG)"; \
	    echo "backend=$(BACKEND)"; \
	    echo "git_commit=$$commit"; echo "git_dirty=$$dirty"; \
	    [ -n "$$ddiff" ] && echo "dirty_diff_sha256=$$ddiff"; \
	    echo "git_untracked=$$untracked"; \
	    [ -n "$$uhash" ] && echo "untracked_tree_sha256=$$uhash"; \
	    echo "audiolib_git_commit=$$ac_commit"; echo "audiolib_git_dirty=$$ac_dirty"; \
	    [ -n "$$ac_ddiff" ] && echo "audiolib_dirty_diff_sha256=$$ac_ddiff"; \
	    echo "audiolib_git_untracked=$$ac_untracked"; \
	    [ -n "$$ac_uhash" ] && echo "audiolib_untracked_tree_sha256=$$ac_uhash"; \
	    echo "aec_git_commit=$$aec_commit"; echo "aec_git_dirty=$$aec_dirty"; \
	    [ -n "$$aec_ddiff" ] && echo "aec_dirty_diff_sha256=$$aec_ddiff"; \
	    echo "aec_git_untracked=$$aec_untracked"; \
	    [ -n "$$aec_uhash" ] && echo "aec_untracked_tree_sha256=$$aec_uhash"; \
	    echo "nr_git_commit=$$nr_commit"; echo "nr_git_dirty=$$nr_dirty"; \
	    [ -n "$$nr_ddiff" ] && echo "nr_dirty_diff_sha256=$$nr_ddiff"; \
	    echo "nr_git_untracked=$$nr_untracked"; \
	    [ -n "$$nr_uhash" ] && echo "nr_untracked_tree_sha256=$$nr_uhash"; \
	    echo "date_utc=$$dateutc"; \
	    echo "allow_dirty_publish=$$adp"; \
	    echo "allow_untracked_publish=$$aup"; \
	  } > "$$work/attest.txt"; \
	  rc=0; "$$work/swapln" --excl-install "$$work/attest.txt" "$$rel_dir/ATTEST/$$attest_name" || rc=$$?; \
	  [ "$$rc" -eq 0 ] && break; \
	  [ "$$rc" -eq 2 ] || { echo "FATAL: could not install attestation $$attest_name" >&2; exit 1; }; \
	  n=$$((n+1)); \
	  [ "$$n" -le 999 ] || { echo "FATAL: 999 attestation name collisions for $$stamp-$$commit (runaway?)" >&2; exit 1; }; \
	done; \
	"$$work/swapln" "$$rel_id" "$(DIST_BACKEND_DIR)/current"; \
	echo "pipelines: published $(BACKEND)/$$rel_id -> $(DIST_BACKEND_DIR)/current (attested: $$attest_name)"
endif

# clean: this dir's hash-keyed obj/bin trees (every config, not just the one this invocation
# resolves to). OBJ_ROOT/BIN_ROOT overrides scrub that tree instead of the real obj/ and bin/. dist/
# is not removed (publish is a deliberate release, not a build cache).
clean:
	rm -rf $(OBJ_ROOT) $(BIN_ROOT)
	rm -f aec_nr_pipeline aec_nr_pipeline_static libaudio_pipeline_dfn.a test_mono_chain

# clean-all: this dir's artifacts plus all three submodule/shared build trees (lib/aec, lib/nr,
# audiolib). clean-libs is an alias with the same effect.
clean-all: clean
	$(MAKE) -C $(AEC_DIR) clean
	$(MAKE) -C $(NR_DIR) clean
	$(MAKE) -C $(COMMON_DIR) clean

clean-libs: clean-all

.PHONY: clean clean-all clean-libs

# --- Header dependency tracking ----------------------------------------------
# See audiolib's Makefile for the full rationale (-MD not -MMD, -MP
# guards against deleted-header stale-.d errors). $(wildcard ...) makes this
# evaluate to nothing (not an error) before any objects exist yet for this
# CFG_SIG.
-include $(wildcard $(OBJ_DIR)/*.d)

# ---------------------------------------------------------------------------
# test-grids: runs both Align-ULCNet deployment grids in one command (`test` only exercises the grid
# ULCNET_GRID selects). ULCNET_GRID is in CFG_SIG, so each grid has its own obj/bin directory. Defined
# outside the COMMON_LIB conditionals: it needs no producer archive itself and each recursive `test`
# bootstraps its own.
# ---------------------------------------------------------------------------
.PHONY: test-grids
test-grids:
	$(MAKE) ULCNET_GRID=16k test
	$(MAKE) ULCNET_GRID=48k test

# test-pairs: the DFN C tests and the mono DFN2 program rebuilt and run for other lookahead pairs
# (DFN_TEST_PAIRS, mask/df), each pair in its own CFG_SIG directories; `make test` is the shipped
# pair. The Python stage and its C-parity shims run their pairs under pytest (dfn/tests). The
# 4-channel DFN units take the same DFN2_MASK_LOOKAHEAD / DFN2_DF_LOOKAHEAD in their Makefile.
DFN_LEGAL_PAIRS := $(foreach m,0 1 2,$(foreach d,0 1 2 3 4,$(m)/$(d)))
DFN_TEST_PAIRS ?= 0/0 0/1 2/0 2/4
DFN_PAIR_TESTS := test_dfn_res_stage test_dfn_rate_bridge test_dfn_hop test_mono_dfn test_readme_flow
.PHONY: test-pairs
test-pairs:
	$(if $(strip $(DFN_TEST_PAIRS)),,$(error DFN_TEST_PAIRS is empty: name at least one mask/df pair))
	$(if $(filter-out $(DFN_LEGAL_PAIRS),$(DFN_TEST_PAIRS)),$(error DFN_TEST_PAIRS takes mask/df pairs from 0..2/0..4, got '$(DFN_TEST_PAIRS)'))
	+@has_n=0; has_q=0; has_t=0; \
	for w in $(MAKEFLAGS); do \
	  case "$$w" in --) break ;; --*|*=*) continue ;; -n|-q|-t) ;; -*) continue ;; esac; \
	  case "$$w" in *n*) has_n=1 ;; esac; \
	  case "$$w" in *q*) has_q=1 ;; esac; \
	  case "$$w" in *t*) has_t=1 ;; esac; \
	done; \
	if [ "$$has_t" = 1 ]; then echo "test-pairs: -t (touch) is a no-op here -- test-pairs runs tests, nothing was touched"; exit 0; fi; \
	if [ "$$has_q" = 1 ]; then exit 0; fi; \
	if [ "$$has_n" = 1 ]; then \
	  for p in $(DFN_TEST_PAIRS); do \
	    echo "test-pairs: would build and run libaudio_pipeline_dfn.a $(DFN_PAIR_TESTS) mono_aec_dfn_res with DFN2_MASK_LOOKAHEAD=$${p%/*} DFN2_DF_LOOKAHEAD=$${p#*/}"; \
	  done; exit 0; \
	fi; \
	exec $(MAKE) --no-print-directory _test_pairs_run

# The run itself, reached only from test-pairs outside -n/-q/-t (under -q the target is reported
# as not up to date: make exits 1 with nothing run).
.PHONY: _test_pairs_run
_test_pairs_run:
	@for p in $(DFN_TEST_PAIRS); do m=$${p%/*}; d=$${p#*/}; \
	  $(MAKE) --no-print-directory DFN2_MASK_LOOKAHEAD=$$m DFN2_DF_LOOKAHEAD=$$d \
	    libaudio_pipeline_dfn.a $(DFN_PAIR_TESTS) mono_aec_dfn_res >/dev/null || exit 1; \
	  bd="$$($(MAKE) -s --no-print-directory DFN2_MASK_LOOKAHEAD=$$m DFN2_DF_LOOKAHEAD=$$d print-bin-dir)"; \
	  nm "$$bd/libaudio_pipeline_dfn.a" | grep -q "dfn2_lookahead_mask$${m}_df$${d}$$" || \
	    { echo "FAIL: libaudio_pipeline_dfn.a was not built for lookahead pair $$p"; exit 1; }; \
	  for t in $(DFN_PAIR_TESTS) mono_aec_dfn_res; do \
	    "$$bd/$$t" >/dev/null || { echo "FAIL: $$t for lookahead pair $$p"; exit 1; }; \
	  done; \
	  echo "lookahead pair $$p: DFN C tests PASS"; \
	done

# pyc-parity: Python <-> C parity of the pipelines glue (mono and 4-channel NR gain, post-beam RES
# gain, shared-delay realign, min/apply/comfort noise/OLA): Layer 1 against std-math C and Layer 2
# shipped against std-math C. Needs python3 with numpy, scipy, soundfile and pyyaml (PYTHON=... to
# choose it); scratch and results go to PYC_WORK (default under $TMPDIR). Options and the exit
# contract are at the top of tests/pyc_parity_pipelines.sh; pass them via PYC_PARITY_ARGS, e.g.
#   make pyc-parity PYC_PARITY_ARGS="--layers 1 --only a.nr4."
# Defined outside the COMMON_LIB conditionals: the script builds its own copies in the scratch dir.
.PHONY: pyc-parity
pyc-parity:
	bash tests/pyc_parity_pipelines.sh $(PYC_PARITY_ARGS)

# ---------------------------------------------------------------------------
# check-pins: verify the producer revisions this checkout expects. lib/aec and lib/nr are submodules
# whose gitlinks are the pin (read from the index, so a staged bump validates before its commit);
# audiolib is a sibling checkout resolved through COMMON_DIR, with its expected commit in audiolib.pin
# (bump it with the gitlinks). A commit mismatch fails; tracked changes in a producer are reported but
# allowed (`make publish` applies its own dirty policy). Publish runs this after taking its lock and
# before building.
# ---------------------------------------------------------------------------
.PHONY: check-pins _check_pins_impl
check-pins: _check_pins_impl

_check_pins_impl:
	@pin="$$(sed -n 's/^AUDIOLIB_COMMIT=//p' audiolib.pin)"; \
	 [ -n "$$pin" ] || { echo "check-pins: audiolib.pin has no AUDIOLIB_COMMIT" >&2; exit 1; }; \
	 have="$$(git -C "$(AUDIOLIB_DIR)" rev-parse HEAD 2>/dev/null || echo unknown)"; \
	 if [ "$$have" != "$$pin" ]; then \
	   echo "check-pins: FAIL audiolib at $(AUDIOLIB_DIR) is $$have, audiolib.pin expects $$pin" >&2; exit 1; \
	 fi; \
	 note=""; [ -n "$$(git -C "$(AUDIOLIB_DIR)" status --porcelain -uno 2>/dev/null)" ] && note=" (tracked changes present)"; \
	 echo "check-pins: audiolib $$have matches audiolib.pin$$note"; \
	 for sub in lib/aec lib/nr; do \
	   rec="$$(git -C .. ls-files -s -- $$sub 2>/dev/null | awk '$$3 == 0 {print $$2}')"; \
	   have="$$(git -C ../$$sub rev-parse HEAD 2>/dev/null || echo unknown)"; \
	   if [ "$$rec" != "$$have" ]; then \
	     echo "check-pins: FAIL $$sub checkout is $$have, gitlink records $$rec" >&2; exit 1; \
	   fi; \
	   note=""; [ -n "$$(git -C ../$$sub status --porcelain -uno 2>/dev/null)" ] && note=" (tracked changes present)"; \
	   echo "check-pins: $$sub $$have matches gitlink$$note"; \
	 done

-include $(wildcard $(OBJ_DIR)/target_abi/*.d)
