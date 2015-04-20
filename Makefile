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
INCFLAGS+=-I$(LIBDIR)/glog/include
INCFLAGS+=-I$(HOME)/tool/openmpi/include

# link flags
LDFLAGS+=-lgfortran -lpthread -lopenblas -lm -fopenmp -glog #-lmpi -lmpi_cxx
LDFLAGS+=-L$(LIBDIR) -L$(LIBDIR)/openblas/lib -L$(LIBDIR)/glog/lib
LDFLAGS+=-L$(HOME)/tool/openmpi/lib

# vpath
VPATH = $(SRCDIR) \
	$(SRCDIR)/config \
	$(SRCDIR)/helper \
	$(SRCDIR)/sgd \
	$(SRCDIR)/layer \
	$(SRCDIR)/connection \
	$(SRCDIR)/network \

# src files
SRCS=\
	$(SRCDIR)/translator.cpp \
	$(SRCDIR)/config/chameleon.cpp \
	$(SRCDIR)/config/configfile.cpp \
	$(SRCDIR)/config/confreader.cpp \
	$(SRCDIR)/helper/matrix.cpp \
	$(SRCDIR)/helper/nonlinearity.cpp \
	$(SRCDIR)/sgd/sgd.cpp \
	$(SRCDIR)/sgd/adagrad.cpp \
	$(SRCDIR)/sgd/adadelta.cpp \
	$(SRCDIR)/sgd/rmsprop.cpp \
	$(SRCDIR)/layer/layer.cpp \
	$(SRCDIR)/layer/input_layer.cpp \
	$(SRCDIR)/layer/lstm_layer.cpp \
	$(SRCDIR)/layer/softmax_layer.cpp \
	$(SRCDIR)/layer/mse_layer.cpp \
	$(SRCDIR)/connection/connection.cpp \
	$(SRCDIR)/network/lstm_rnn.cpp \
	$(SRCDIR)/network/rnn_translator.cpp

# obj files using patsubst matching
OBJS=$(SRCS:%.cpp=%.o)

# nothing to do here
# .PHONY: 

# all comes first in the file, so it is the default 
# all : lstmRNN
all : RNNTranslator

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