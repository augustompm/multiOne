all: bali_score bali_score_block bali_score_reliable

clean:
	rm -f *.o bali_score bali_score_block bali_score_reliable

HEADERS = general.h clustalw.h

CC	= cc
CFLAGS  = -c -g -I$(EXPAT_INC)
LFLAGS	= -g -lm -L$(EXPAT_LIB) -lexpat

# Ajuste os caminhos do expat conforme necessário
EXPAT_LIB	= /usr/lib
EXPAT_INC	= /usr/include

bali_score : readseq.o readxml.o init.o util.o bali_score.o
	$(CC) -o $@ readseq.o readxml.o init.o util.o bali_score.o $(LFLAGS)

bali_score_block : readseq.o readxml.o init.o util.o bali_score_block.o
	$(CC) -o $@ readseq.o readxml.o init.o util.o bali_score_block.o $(LFLAGS)

bali_score_reliable : readseq.o readxml.o init.o util.o bali_score_reliable.o
	$(CC) -o $@ readseq.o readxml.o init.o util.o bali_score_reliable.o $(LFLAGS)

.c.o : $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<