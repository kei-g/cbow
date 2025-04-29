BINARY=bin/cbow
CXX:=clang++
CXXFLAGS+=-D_DEFAULT_SOURCE
CXXFLAGS+=-Oz
CXXFLAGS+=-Wall
CXXFLAGS+=-Werror
CXXFLAGS+=-Wextra
CXXFLAGS+=-c
CXXFLAGS+=-faddrsig
CXXFLAGS+=-fdata-sections
#CXXFLAGS+=-ffast-math
CXXFLAGS+=-ffunction-sections
CXXFLAGS+=-fintegrated-as
CXXFLAGS+=-fintegrated-cc1
CXXFLAGS+=-flto=full
#CXXFLAGS+=-fno-exceptions
CXXFLAGS+=-fno-stack-protector
CXXFLAGS+=-fno-threadsafe-statics
CXXFLAGS+=-fpic
CXXFLAGS+=-fpie
CXXFLAGS+=-fno-plt
CXXFLAGS+=-iquote$(shell pwd)/inc
CXXFLAGS+=-march=native
CXXFLAGS+=-pedantic
CXXFLAGS+=-std=c++2c
CXXFLAGS+=-stdlib=libc++
HEADERS=$(call wildcard_r,inc,*.hh)
LD:=clang++
LDFLAGS+=-Wl,--build-id=fast
LDFLAGS+=-Wl,--gc-sections
LDFLAGS+=-Wl,--icf=all
LDFLAGS+=-Wl,--lto-O3
LDFLAGS+=-Wl,--no-eh-frame-hdr
LDFLAGS+=-Wl,--no-gnu-unique
LDFLAGS+=-Wl,--pie
LDFLAGS+=-Wl,-z,noexecstack
LDFLAGS+=-Wl,-z,relro,-z,now
LDFLAGS+=-fuse-ld=lld
LDFLAGS+=-rtlib=compiler-rt
LDFLAGS+=-stdlib=libc++
LDFLAGS+=-unwindlib=libunwind
OBJECTS=$(SOURCES:%.cc=%.o)
SOURCES=$(call wildcard_r,src,*.cc)

wildcard_r=$(foreach d,$(wildcard $(1:=/*)),$(call wildcard_r,$d,$2) $(filter $(subst *,%,$2),$d))

all: $(BINARY)

clean:
	@$(RM) -v $(BINARY) $(OBJECTS)

.PHONY: all clean

.cc.o:
	$(CXX) $(sort $(CXXFLAGS)) -o $(<:%.cc=%.o) $<

bin:
	@mkdir -v $@

$(BINARY): bin $(OBJECTS)
	$(LD) $(sort $(LDFLAGS)) -o $@ $(OBJECTS)
	@llvm-strip --strip-all $@

$(OBJECTS): $(HEADERS) Makefile
