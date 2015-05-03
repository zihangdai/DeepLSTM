#include "master.h"
#include "rnn_translator.h"

sgdBase * initSgdSolver (boost::property_tree::ptree *confReader, string section, int paramSize) {
	int solverType = confReader->get<int>(section+"solver_type");
	sgdBase *sgdSolver;
	switch (solverType) {
		// sgdBasic
		case 0: {
			printf("Init basic sgd solver.\n");
			sgdSolver = new sgdBasic(confReader, section, paramSize);
			break;
		}
		// adagrad
		case 1: {
			printf("Init adagrad solver.\n");
			sgdSolver = new adagrad(confReader, section, paramSize);
			break;
		}
		// adadelta
		case 2: {
			sgdSolver = new adadelta(confReader, section, paramSize);
			printf("Init adadelta solver.\n");
			break;
		}
		// rmsprop
		case 3: {
			sgdSolver = new rmsprop(confReader, section, paramSize);
			printf("Init rmsprop solver.\n");
			break;
		}
		default: {
			printf("Error solver type.\n");
			exit(-1);
		}
	}
	return sgdSolver;
}

void masterFunc () {
	/****************************************************************
	* Step 1: Setup and Initialization
	* Load conf, init model, allocate mem, init params, init solver
	* Load cross-validation data
	****************************************************************/

	// Step 1.1: Load configuration
	boost::property_tree::ptree *confReader = new boost::property_tree::ptree();
	boost::property_tree::ini_parser::read_ini("mpi_translator.conf", *confReader);
	
	string section = "Master.";
	int validBatchSize = confReader->get<int>(section + "validation_batch_size");
	int nSendMax = confReader->get<int>(section + "max_iteration_number");

	// Step 1.2 Initialize model
	openblas_set_num_threads(1);
	section = "Translator.";
	int max_openmp_threads = confReader->get<int>(section + "max_threads");
	omp_set_num_threads(max_openmp_threads);	
	RNNTranslator *translator = new RNNTranslator(confReader, section);

	int paramSize = translator->m_paramSize;
	printf("paramSize: %d\n", paramSize);

	// Step 1.3: Allocate master memory
	float *params = new float[paramSize];
	float *grad = new float[paramSize];

	// Step 1.4: Initialize params
	translator->initParams(params);
	
	// Step 1.5: Initialize SGD Solver
	section = "SGD.";
	sgdBase *sgdSolver = initSgdSolver(confReader, section, paramSize);
	printf("MASTER: finish step 1\n");

	// Step 1.6: Load cross-validation data
	// section = "ValidationData.";
	// DataFactory *dataset = initDataFactory(confReader, section);
	// int numSample = dataset->getNumberOfData();
	// int dataSize  = dataset->getDataSize();
	// int labelSize = dataset->getLabelSize();

	// float *data  = new float[validBatchSize * dataSize];
	// float *label = new float[validBatchSize * labelSize];

	/****************************************************************
	* Step 2: Seed the slaves
	* (1) Broadcast paramSize to all slaves
	* (2) Send the same initial params with WORKTAG to all slaves
	****************************************************************/
	int nProc;
	MPI_Comm_size(MPI_COMM_WORLD, &nProc);
	int nSlave = nProc - 1;

	MPI_Bcast(&paramSize, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
	
	int nSend = 0;
	int nRecv = 0;
	for (int rank = 1; rank < nProc; ++rank) {
		MPI_Send(params, paramSize, MPI_FLOAT, rank, WORKTAG, MPI_COMM_WORLD);
		nSend++;
	}
	printf("MASTER: finish step 2\n");

	/****************************************************************
	* Step 3: Paralleled training
	* Receive mini-batch grad from *ANY* slave
	* Update params based received grad
	* Re-send params to slave to process next mini-batch
	****************************************************************/
	
	MPI_Status status;
	// TEMP while loop condition
	while (nSend < nSendMax) {
		MPI_Recv(grad, paramSize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		nRecv++;
		
		sgdSolver->updateParams(params, grad, status.MPI_SOURCE);
		
		// Send updated params to corresponding slave
		MPI_Send(params, paramSize, MPI_FLOAT, status.MPI_SOURCE, WORKTAG, MPI_COMM_WORLD);
		nSend++;
	}
	printf("MASTER: finish step 3\n");
	
	/****************************************************************
	* Step 4: Stop the slaves
	****************************************************************/
	
	// Step 4.1: Receive all dispatched but irreceived grad result
	while (nRecv < nSend) {
		MPI_Recv(grad, paramSize, MPI_FLOAT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		sgdSolver->updateParams(params, grad, status.MPI_SOURCE);

		nRecv++;
	}
	// Step 4.2: Send STOPTAG to all slaves
	for (int rank = 1; rank < nProc; ++rank) {
		MPI_Send(&rank, 1, MPI_INT, rank, STOPTAG, MPI_COMM_WORLD);
	}
	printf("MASTER: finish step 4\n");
	
	/****************************************************************
	* Step 5: Save trained parameters and clear things
	****************************************************************/
	section = "Master.";
	string saveFilename = confReader->get<string>(section + "save_filename");

	ofstream savefile (saveFilename.c_str(), ios::out|ios::binary);
	if (savefile.is_open()) {
		savefile.write ((char *)params, sizeof(float) * paramSize);
		savefile.close();
	} else {
		printf("Failed to open savefile\n");
		exit(1);
	}

	delete sgdSolver;
	delete translator;
	delete [] params;
	delete [] grad;
}