all: main;

clean:
	rm -f main main.o	
	
main: main.o
	g++ -std=c++11 -o main main.o

main.o: main.cpp 
	g++ -std=c++11 -c main.cpp
