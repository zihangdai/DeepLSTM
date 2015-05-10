#include "slave.h"
#include "rnn_lstm.h"

using namespace std;

DataFactory* initDataFactory(boost::property_tree::ptree *confReader, string section)
{
	int dataIndex = confReader->get<int>(section + "data_type");
	DataFactory* data;
	switch(dataIndex) {
		case 0: {
			printf("Slave Data: Init Sequence Data.\n");
			data = new SequenceData(confReader, section);
			break;
		} 
		case 1: {
			printf("Slave Data: Init MNIST Data.\n");
			data = new Mnist(confReader, section);
			break;
		}
		default:  {
			printf("Error, no Data Index");
			exit(-1);
		}
	}
	return data;
}



void slaveFunc(int argc, char ** argv){ 
	if (argc < 2) {
		printf("argc %d\n", argc);
		exit(1);
	}
	string dirPath = argv[1];

	// MPI Setup
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	srand(time(NULL) * rank);
	
	// step 0: Init conf and read basic slave conf
	boost::property_tree::ptree *confReader = new boost::property_tree::ptree();
	boost::property_tree::ini_parser::read_ini(dirPath+"mpi.conf", *confReader);
	
	string section = "Slave.";
	int batchSize = confReader->get<int>(section + "training_batch_size");

	// step 1: Init model and allocate related memorys
	section = "LSTM.";
	
	openblas_set_num_threads(1);
	int max_openmp_threads = confReader->get<int>(section + "max_threads");
	omp_set_num_threads(max_openmp_threads);
	omp_set_nested(0);
	printf("SLAVE[%d]: openmp threads: max threads %d, nested %d\n", rank, omp_get_max_threads(), omp_get_nested());	

	RecurrentNN *rnn = new RNNLSTM(confReader, section);
	int paramSize;
	MPI_Bcast(&paramSize, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

	float *param = new float[paramSize];
	float *grad  = new float[paramSize];

	// step 2: Init Data and allocate related memorys
	section = "Data.";
	DataFactory *dataset = initDataFactory(confReader, section);
	int numSample = dataset->getNumberOfData();
	int dataSize  = dataset->getDataSize();
	int labelSize = dataset->getLabelSize();

	float *data  = new float[batchSize * dataSize];
	float *label = new float[batchSize * labelSize];

	int *indices = new int[numSample];
	int *pickIndices = new int[batchSize];
	for (int i=0;i<numSample;i++){
		indices[i]=i;
	}
	std::random_shuffle(indices, indices + numSample);
	
	// step 3: Begin Training taks
	int index = 0, taskCount=0;
	MPI_Status status;
	printf("Slave[%d] go into training loop\n", rank);
	while (1) {
		/*step 3.1:receive from master*/
		MPI_Recv(param, paramSize, MPI_FLOAT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		
		/*step 3.2: check whether ends*/
		if(status.MPI_TAG == STOPTAG){
			break;
		} 
		
		/*step 3.3: request for data*/
		if (index + batchSize >= numSample){
			std::random_shuffle(indices, indices + numSample);
			index = 0;
		}
		for(int i=0;i<batchSize;i++){
			pickIndices[i] = indices[index];
			index++;
		}
		dataset->getDataBatch(label, data, pickIndices, batchSize);

		/*step 3.4: calculate the grad*/
		double begTime = CycleTimer::currentSeconds();
		float error = rnn->computeGrad(grad, param, data, label, batchSize);
		double endTime = CycleTimer::currentSeconds();
		printf("Slave[%d] finished task %d: error %f, time %f\n", rank, ++taskCount, error, endTime - begTime);
		
		/*step 3.5: return to master*/
		MPI_Send(grad, paramSize, MPI_FLOAT, ROOT, rank, MPI_COMM_WORLD);
	}

	delete [] param;
	delete [] grad;
	
	delete [] label;
	delete [] data;
	
	delete [] indices;
	delete [] pickIndices;

	delete confReader;
	delete rnn;
	delete dataset;
}