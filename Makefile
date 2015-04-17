# dirs
OBJDIR=objs
SRCDIR=src
LIBDIR=lib

# compiler
CXX=g++

# compile flags
CXXFLAGS+=-O3 -m64 -mavx

# include flags
INCFLAGS+=$(foreach d, $(VPATH), -I$d)
INCFLAGS+=-I$(LIBDIR)/openblas/include

# link flags
LDFLAGS+=-lgfortran -lpthread -lopenblas -lm -fopenmp #-lmpi -lmpi_cxx
LDFLAGS+=-L$(LIBDIR) -L$(LIBDIR)/openblas/lib

# vpath
VPATH = $(SRCDIR) \
	$(SRCDIR)/config \
	$(SRCDIR)/connection \
	$(SRCDIR)/helper \
	$(SRCDIR)/layer \
	$(SRCDIR)/network \

# src files
SRCS=\
	$(SRCDIR)/translator.cpp \
	$(SRCDIR)/config/chameleon.cpp \
	$(SRCDIR)/config/configfile.cpp \
	$(SRCDIR)/config/confreader.cpp \
	$(SRCDIR)/connection/connection.cpp \
	$(SRCDIR)/helper/matrix.cpp \
	$(SRCDIR)/helper/nonlinearity.cpp \
	$(SRCDIR)/layer/layer.cpp \
	$(SRCDIR)/layer/input_layer.cpp \
	$(SRCDIR)/layer/lstm_layer.cpp \
	$(SRCDIR)/layer/softmax_layer.cpp \
	$(SRCDIR)/layer/mse_layer.cpp \
	$(SRCDIR)/network/lstm_rnn.cpp \
	$(SRCDIR)/network/rnn_translator.cpp

# obj files using patsubst matching
OBJS=$(SRCS:%.cpp=%.o)

# nothing to do here
# .PHONY: 

# all comes first in the file, so it is the default 
all : lstmRNN

# compile main program parallelSGD from all objs 
# lstmRNN: $(OBJS)
RNNTranslator: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCFLAGS) $(LDFLAGS) $^ -o $@

# order-only prerequisites for OBJDIR
$(OBJS): | $(OBJDIR)
$(OBJDIR):
	mkdir -p $@

# compile all objs from corresponding %.cpp file and all other *.h files
%.o: %.cpp 
	$(CXX) $(CXXFLAGS) $(INCFLAGS) $(LDFLAGS) $< -c -o $@

# clean
clean:
	rm -rf $(OBJS) lstmRNN