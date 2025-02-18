#include "muscle.h"

extern float Blosum62_sij[20][20];
//// Float matrix 0.5 bit units, 2 decimals, C source
//
//float B62Mf[20][20] = {
////        A       C       D       E       F       G       H       I       K       L       M       N       P       Q       R       S       T       V       W       Y
//  {   3.93f, -0.41f, -1.75f, -0.86f, -2.21f,  0.16f, -1.63f, -1.32f, -0.73f, -1.46f, -0.94f, -1.53f, -0.81f, -0.80f, -1.41f,  1.12f, -0.05f, -0.19f, -2.53f, -1.76f,  }, // A 
//  {  -0.41f,  8.58f, -3.46f, -3.61f, -2.38f, -2.50f, -2.99f, -1.23f, -3.04f, -1.28f, -1.42f, -2.66f, -2.80f, -2.90f, -3.39f, -0.88f, -0.87f, -0.81f, -2.30f, -2.41f,  }, // C 
//  {  -1.75f, -3.46f,  5.77f,  1.51f, -3.48f, -1.31f, -1.12f, -3.12f, -0.70f, -3.61f, -3.06f,  1.27f, -1.48f, -0.31f, -1.61f, -0.26f, -1.05f, -3.14f, -4.21f, -3.06f,  }, // D 
//  {  -0.86f, -3.61f,  1.51f,  4.90f, -3.19f, -2.11f, -0.12f, -3.19f,  0.78f, -2.85f, -2.00f, -0.27f, -1.12f,  1.85f, -0.12f, -0.15f, -0.86f, -2.44f, -2.84f, -2.02f,  }, // E 
//  {  -2.21f, -2.38f, -3.48f, -3.19f,  6.05f, -3.11f, -1.23f, -0.16f, -3.08f,  0.41f,  0.01f, -2.99f, -3.60f, -3.16f, -2.79f, -2.37f, -2.11f, -0.85f,  0.92f,  2.94f,  }, // F 
//  {   0.16f, -2.50f, -1.31f, -2.11f, -3.11f,  5.56f, -2.04f, -3.72f, -1.53f, -3.63f, -2.68f, -0.42f, -2.13f, -1.79f, -2.30f, -0.29f, -1.58f, -3.14f, -2.49f, -3.04f,  }, // G 
//  {  -1.63f, -2.99f, -1.12f, -0.12f, -1.23f, -2.04f,  7.51f, -3.23f, -0.72f, -2.79f, -1.55f,  0.58f, -2.16f,  0.45f, -0.25f, -0.88f, -1.69f, -3.12f, -2.34f,  1.69f,  }, // H 
//  {  -1.32f, -1.23f, -3.12f, -3.19f, -0.16f, -3.72f, -3.23f,  4.00f, -2.67f,  1.52f,  1.13f, -3.22f, -2.76f, -2.77f, -2.99f, -2.35f, -0.72f,  2.55f, -2.58f, -1.33f,  }, // I 
//  {  -0.73f, -3.04f, -0.70f,  0.78f, -3.08f, -1.53f, -0.72f, -2.67f,  4.50f, -2.45f, -1.35f, -0.18f, -1.01f,  1.27f,  2.11f, -0.20f, -0.67f, -2.26f, -2.96f, -1.82f,  }, // K 
//  {  -1.46f, -1.28f, -3.61f, -2.85f,  0.41f, -3.63f, -2.79f,  1.52f, -2.45f,  3.85f,  1.99f, -3.38f, -2.86f, -2.13f, -2.15f, -2.44f, -1.20f,  0.79f, -1.63f, -1.06f,  }, // L 
//  {  -0.94f, -1.42f, -3.06f, -2.00f,  0.01f, -2.68f, -1.55f,  1.13f, -1.35f,  1.99f,  5.39f, -2.15f, -2.48f, -0.42f, -1.37f, -1.48f, -0.67f,  0.69f, -1.42f, -0.99f,  }, // M 
//  {  -1.53f, -2.66f,  1.27f, -0.27f, -2.99f, -0.42f,  0.58f, -3.22f, -0.18f, -3.38f, -2.15f,  5.65f, -2.00f,  0.00f, -0.44f,  0.60f, -0.05f, -2.88f, -3.70f, -2.08f,  }, // N 
//  {  -0.81f, -2.80f, -1.48f, -1.12f, -3.60f, -2.13f, -2.16f, -2.76f, -1.01f, -2.86f, -2.48f, -2.00f,  7.36f, -1.28f, -2.11f, -0.81f, -1.08f, -2.35f, -3.65f, -2.92f,  }, // P 
//  {  -0.80f, -2.90f, -0.31f,  1.85f, -3.16f, -1.79f,  0.45f, -2.77f,  1.27f, -2.13f, -0.42f,  0.00f, -1.28f,  5.29f,  0.98f, -0.10f, -0.68f, -2.20f, -1.95f, -1.42f,  }, // Q 
//  {  -1.41f, -3.39f, -1.61f, -0.12f, -2.79f, -2.30f, -0.25f, -2.99f,  2.11f, -2.15f, -1.37f, -0.44f, -2.11f,  0.98f,  5.47f, -0.76f, -1.12f, -2.50f, -2.68f, -1.69f,  }, // R 
//  {   1.12f, -0.88f, -0.26f, -0.15f, -2.37f, -0.29f, -0.88f, -2.35f, -0.20f, -2.44f, -1.48f,  0.60f, -0.81f, -0.10f, -0.76f,  3.88f,  1.38f, -1.65f, -2.75f, -1.69f,  }, // S 
//  {  -0.05f, -0.87f, -1.05f, -0.86f, -2.11f, -1.58f, -1.69f, -0.72f, -0.67f, -1.20f, -0.67f, -0.05f, -1.08f, -0.68f, -1.12f,  1.38f,  4.55f, -0.06f, -2.43f, -1.61f,  }, // T 
//  {  -0.19f, -0.81f, -3.14f, -2.44f, -0.85f, -3.14f, -3.12f,  2.55f, -2.26f,  0.79f,  0.69f, -2.88f, -2.35f, -2.20f, -2.50f, -1.65f, -0.06f,  3.77f, -2.83f, -1.21f,  }, // V 
//  {  -2.53f, -2.30f, -4.21f, -2.84f,  0.92f, -2.49f, -2.34f, -2.58f, -2.96f, -1.63f, -1.42f, -3.70f, -3.65f, -1.95f, -2.68f, -2.75f, -2.43f, -2.83f, 10.50f,  2.15f,  }, // W 
//  {  -1.76f, -2.41f, -3.06f, -2.02f,  2.94f, -3.04f,  1.69f, -1.33f, -1.82f, -1.06f, -0.99f, -2.08f, -2.92f, -1.42f, -1.69f, -1.69f, -1.61f, -1.21f,  2.15f,  6.59f,  }, // Y 
//};

void MakeBlosum62SMx(const Sequence &A, const Sequence &B, Mx<float> &MxS)
	{
	uint LA = A.GetLength();
	uint LB = B.GetLength();
	const char *ptrA = A.GetCharPtr();
	const char *ptrB = B.GetCharPtr();
	MxS.Alloc("BlosumS", LA, LB);
	float **S = MxS.GetData();
	for (uint i = 0; i < LA; ++i)
		{
		char a = ptrA[i];
		byte ai = g_CharToLetterAmino[a];
		for (uint j = 0; j < LB; ++j)
			{
			char b = ptrB[j];
			byte bi = g_CharToLetterAmino[b];
			if (ai < 20 && bi < 20)
				S[i][j] = Blosum62_sij[ai][bi];
			else
				S[i][j] = 0;
			}
		}
	}

void MakeBlosum62SMx(const string &A, const string &B, Mx<float> &MxS)
	{
	float GetBlosumScoreChars(byte a, byte b);

	uint LA = SIZE(A);
	uint LB = SIZE(B);
	MxS.Alloc("BlosumS", LA, LB);
	float **S = MxS.GetData();
	for (uint i = 0; i < LA; ++i)
		{
		char a = A[i];
		for (uint j = 0; j < LB; ++j)
			{
			char b = B[j];
			S[i][j] = GetBlosumScoreChars(a, b);
			}
		}
	}

void GetBlosum62LogOddsLetterMx(vector<vector<float> > &LogOddsMx)
	{
	LogOddsMx.clear();
	LogOddsMx.resize(20);
	for (uint i = 0; i < 20; ++i)
		LogOddsMx[i].resize(20);
	for (uint i = 0; i < 20; ++i)
		for (uint j = 0; j < 20; ++j)
			LogOddsMx[i][j] = Blosum62_sij[i][j];
	}
