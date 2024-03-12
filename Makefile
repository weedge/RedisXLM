ROOT=.

include $(ROOT)/deps/readies/mk/main

#----------------------------------------------------------------------------------------------

define HELPTEXT
make build_hnsw		# build examples
make build
  RELEASE=1         # build release variant

make clean         # remove binary files
  ALL=1            # remove binary directories

make all           # build all libraries and packages

make test          # run tests

make info          # show toolchain version


endef

#----------------------------------------------------------------------------------------------

MK_CUSTOM_CLEAN=1
BINDIR=$(BINROOT)

include $(MK)/defs
include $(MK)/rules

#----------------------------------------------------------------------------------------------

ifeq ($(RELEASE),1)
CARGO_FLAGS += --release
TARGET_DIR=target/release
else
TARGET_DIR=target/debug
endif

#----------------------------------------------------------------------------------------------

lint:
	cargo fmt -- --check

.PHONY: lint

#----------------------------------------------------------------------------------------------

RUST_SOEXT.linux=so
RUST_SOEXT.freebsd=so
RUST_SOEXT.macos=dylib

build:
	cargo build --all --all-targets $(CARGO_FLAGS)

build_example:
	cargo build --examples

build_llamacpp:
	cargo build --manifest-path rust/llamacpp/Cargo.toml $(CARGO_FLAGS)

clean:
ifneq ($(ALL),1)
	cargo clean
else
	rm -rf target
endif

.PHONY: build clean

#----------------------------------------------------------------------------------------------

test: cargo_test_workspace

cargo_test_workspace: build
	cargo test --workspace \
		$(CARGO_FLAGS)

cargo_test: build
	cargo test --tests $(CARGO_FLAGS)

cargo_test_doc:
	cargo test --doc --workspace $(CARGO_FLAGS)

.PHONY: test cargo_test cargo_test_workspace cargo_test_doc

#----------------------------------------------------------------------------------------------

info:
	gcc --version
	cmake --version
	clang --version
	rustc --version
	cargo --version
	rustup --version
	rustup show

.PHONY: info