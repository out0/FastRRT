all: bin

bin:
	rm -rf build
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)
	mv build/test .
	rm -rf build

test: bin
	./test
	rm test

clean:
	rm -rf build test

