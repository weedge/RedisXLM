ROOT=.

include $(ROOT)/deps/readies/mk/main

#----------------------------------------------------------------------------------------------

define HELPTEXT
make build_hnsw		# build hnsw crate
make build
  RELEASE=1          # build release variant

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
	export LIBRARY_PATH=/usr/local/lib && RUSTFLAGS="-C link-args=-Wl,-rpath,/usr/local/lib" cargo build --all --all-targets $(CARGO_FLAGS)

build_example:
	export LIBRARY_PATH=/usr/local/lib && RUSTFLAGS="-C link-args=-Wl,-rpath,/usr/local/lib" cargo build --examples

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
	export LIBRARY_PATH=/usr/local/lib && RUSTFLAGS="-C link-args=-Wl,-rpath,/usr/local/lib" cargo test --workspace \
		$(CARGO_FLAGS)

cargo_test: build
	export LIBRARY_PATH=/usr/local/lib && RUSTFLAGS="-C link-args=-Wl,-rpath,/usr/local/lib" cargo test --tests $(CARGO_FLAGS)

cargo_test_doc:
	export LIBRARY_PATH=/usr/local/lib && RUSTFLAGS="-C link-args=-Wl,-rpath,/usr/local/lib" cargo test --doc --workspace $(CARGO_FLAGS)

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