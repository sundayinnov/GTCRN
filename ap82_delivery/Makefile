CC = gcc
CFLAGS = -Wall -O2 -g -fsanitize=address
TARGET = test_gtcrn
SRCS = main.c gtcrn.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) -lm

clean:
	rm -f $(TARGET)
