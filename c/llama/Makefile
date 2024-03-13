# ReidsXSlot Makefile
# Copyright (C) 2024- weedge <weege007 at gmail dot com>
# This file is released under the MIT license, see the LICENSE file

CC=gcc
BUILD_TYPE ?= Debug
# redis version >= 5.0.0
REDIS_VERSION ?= 50000

# REDISXLM_LLAMA version
REDISXLM_LLAMA_MAJOR=$(shell grep REDISXLM_LLAMA_MAJOR REDISXLM_LLAMA.h | awk '{print $$3}')
REDISXLM_LLAMA_MINOR=$(shell grep REDISXLM_LLAMA_MINOR REDISXLM_LLAMA.h | awk '{print $$3}')
REDISXLM_LLAMA_PATCH=$(shell grep REDISXLM_LLAMA_PATCH REDISXLM_LLAMA.h | awk '{print $$3}')
REDISXLM_LLAMA_SONAME=$(shell grep REDISXLM_LLAMA_SONAME REDISXLM_LLAMA.h | awk '{print $$3}')

SOURCEDIR=$(shell pwd -P)

# RedisModulesSDK
SDK_DIR = ${SOURCEDIR}/../deps/RedisModulesSDK
SDK_CFLAGS ?= -I$(SDK_DIR) -I$(SDK_DIR)/rmutil
SDK_LDFLAGS ?= -L$(SDK_DIR)/rmutil -lrmutil

# threadpool
THREADPOOL_DIR = ${SOURCEDIR}/../deps/threadpool
THREADPOOL_CFLAGS ?= -I$(THREADPOOL_DIR)

# redis data struct
DS_DIR = ${SOURCEDIR}/../deps/ds
DS_CFLAGS ?= -I$(DS_DIR)

#set environment variable RM_INCLUDE_DIR to the location of redismodule.h
ifndef RM_INCLUDE_DIR
	RM_INCLUDE_DIR=$(SDK_DIR)
endif

ENABLE_SANITIZE?=NO
SANITIZE_CFLAGS?=
SANITIZE_LDLAGS?=
OPTIMIZE_CFLAGS?=-O3
ifeq ($(BUILD_TYPE),Debug)
ifeq ($(ENABLE_SANITIZE),YES)
# https://gist.github.com/weedge/bdf786fb9ccdf4d84ba08ae8e71c5f98
# https://github.com/google/sanitizers/issues/679
	SANITIZE_CFLAGS=-fsanitize=address -fno-omit-frame-pointer -fsanitize-address-use-after-scope 
	SANITIZE_LDLAGS=-fsanitize=address -lasan
endif
	OPTIMIZE_CFLAGS=-O1
endif
# find the OS
uname_S := $(shell sh -c 'uname -s 2>/dev/null || echo not')
# Compile flags for linux / osx
ifeq ($(uname_S),Linux)
	SHOBJ_CFLAGS ?= $(OPTIMIZE_CFLAGS) \
					$(SANITIZE_CFLAGS) \
					-DREDIS_VERSION=$(REDIS_VERSION) -I$(RM_INCLUDE_DIR) \
					-fPIC -W -Wall -fno-common -g -ggdb -std=gnu99 \
					-D_GNU_SOURCE -D_XOPEN_SOURCE=600 \
					-pthread -fvisibility=hidden 
	SHOBJ_LDFLAGS ?= -fPIC -shared -Bsymbolic \
					$(SANITIZE_LDLAGS) \
					-fvisibility=hidden
					
else
	SHOBJ_CFLAGS ?= $(OPTIMIZE_CFLAGS) \
					$(SANITIZE_CFLAGS) \
					-DREDIS_VERSION=$(REDIS_VERSION) -I$(RM_INCLUDE_DIR) \
					-fPIC -W -Wall -dynamic -fno-common -g -ggdb -std=gnu99 \
					-D_GNU_SOURCE \
					-pthread -fvisibility=hidden
ifeq ($(ENABLE_SANITIZE),YES)
	LD=clang
	SANITIZE_LDLAGS=-fsanitize=address
endif
	SHOBJ_LDFLAGS ?= -bundle -undefined dynamic_lookup \
					$(SANITIZE_LDLAGS) \
					-keep_private_externs
endif

# OS X 11.x doesn't have /usr/lib/libSystem.dylib and needs an explicit setting.
ifeq ($(uname_S),Darwin)
ifeq ("$(wildcard /usr/lib/libSystem.dylib)","")
APPLE_LIBS = -L /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib -lsystem
endif
endif

.SUFFIXES: .c .so .xo .o
CC_SOURCES = $(wildcard $(SOURCEDIR)/*.c) \
	$(wildcard $(THREADPOOL_DIR)/thpool.c) \
	$(wildcard $(DS_DIR)/*.c) 
CC_OBJECTS = $(sort $(patsubst %.c, %.o, $(CC_SOURCES)))


all: init ${THREADPOOL_DIR}/thpool.o redisxlm_llama.so ldd_so

help:
	@echo -e "please choose make with below env params:\n"
	@echo "BUILD_TYPE={Debug or Release} default Debug"
	@echo "ENABLE_SANITIZE={YES or NO} default NO"
	@echo "RM_INCLUDE_DIR={redis_absolute_path}/src, include redismodule.h"
	@echo "REDIS_VERSION=6000, default 6000(6.0.0), use 70200(7.2.0) inlcude 7.2.0+ redismodule.h to use feature api"
	@echo -e "\nhave fun :)"

init:
	@git submodule init
	@git submodule update
	@make -C $(SDK_DIR)/rmutil CFLAGS="-g -fPIC $(OPTIMIZE_CFLAGS) -std=gnu99 -Wall -Wno-unused-function -fvisibility=hidden -I$(RM_INCLUDE_DIR)"

${THREADPOOL_DIR}/thpool.o: ${THREADPOOL_DIR}/thpool.c
	$(CC) -c -o $@ $(SHOBJ_CFLAGS) \
	$<

${SOURCEDIR}/module.o: ${SOURCEDIR}/module.c
	$(CC) -c -o $@ $(SHOBJ_CFLAGS) $(DS_CFLAGS) \
	$(THREADPOOL_CFLAGS) \
	$< 

${SOURCEDIR}/redisxlm_llama.o: ${SOURCEDIR}/REDISXLM_LLAMA.c
	$(CC) -c -o $@ $(SHOBJ_CFLAGS) $(DS_CFLAGS) \
	$(THREADPOOL_CFLAGS) \
	$<

%.o: %.c
	$(CC) -c -o $@ $(SHOBJ_CFLAGS) $<

redisxlm_llama.so: $(CC_OBJECTS)
	$(LD) -o $@ $(CC_OBJECTS) \
	$(SHOBJ_LDFLAGS) \
	$(APPLE_LIBS) \
	-lc

ldd_so:
ifeq ($(uname_S),Darwin)
	@rm -rvf $(SOURCEDIR)/redisxlm_llama.dylib.$(REDISXLM_LLAMA_SONAME)
	@otool -L $(SOURCEDIR)/redisxlm_llama.so
	@ln -s $(SOURCEDIR)/redisxlm_llama.so $(SOURCEDIR)/redisxlm_llama.dylib.$(REDISXLM_LLAMA_SONAME)
else
	@rm -rvf $(SOURCEDIR)/redisxlm_llama.so.$(REDISXLM_LLAMA_SONAME)
	@ldd $(SOURCEDIR)/redisxlm_llama.so
	@ln -s $(SOURCEDIR)/redisxlm_llama.so $(SOURCEDIR)/redisxlm_llama.so.$(REDISXLM_LLAMA_SONAME)
endif

clean:
	cd $(SOURCEDIR) && rm -rvf *.xo *.so *.o *.a
	cd $(DS_DIR) && rm -rvf *.xo *.so *.o *.a
	cd $(THREADPOOL_DIR) && rm -rvf *.xo *.so *.o *.a
	cd $(SDK_DIR)/rmutil && make clean 
	rm -rvf $(SOURCEDIR)/redisxlm_llama.so.$(REDISXLM_LLAMA_SONAME)
	rm -rvf $(SOURCEDIR)/redisxlm_llama.dylib.$(REDISXLM_LLAMA_SONAME)
