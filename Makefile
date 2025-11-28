all: bin

bin:
	rm -rf build
	mkdir -p build
	cd build && cmake ..
	cd build && make -j$(nproc)

local: bin
	cp build/libfastrrt.so.1	./libfastrrt.so
	cp ../libdriveless/python_bind/app/pydriveless/cpp/libdriveless.so ./libdriveless.so
	rm -rf build

install: bin 
	cd build && sudo make install
	cp build/libfastrrt.so.2.0.0	python_bind/app/pyfastrrt/cpp/libfastrrt.so
	cd python_bind && pip3 install . --break-system-packages
	cd python_bind && rm -rf build
	cd python_bind && rm -rf dist	
	rm -rf build

test: bin
	./unittest
	rm unittest

clean:
	rm -rf build unittest
