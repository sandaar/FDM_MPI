#include <sys/time.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define M 81    // Кол-во узлов по ширине
#define N 61    // Кол-во узлов по высоте

struct InitValues{
    double width;
    double height;
    double initTemp;
    double rightTemp;
    double bottomTemp; 
    double left_dTdn;
    double top_dTdn;
} values = {8.0, 6.0, 0.0, 20.0, 100.0, 0.0, 0.0};
struct timeval tv1,tv2,dtv;
struct timezone tz;
double *matrixClone, *matrixCoeff, *matrixTemp;
int *stBl, *lenBl;
int NM,numBl,numr,numt;
double dt,t_stop,t_pause;
void time_start();
long time_stop();
void gnuplotInit();
void drawTemp();
void initMatrix();
void matrixesToFile();
void pryamGauss(double*, double*);
void obratGauss(double*, double*);
void solveMatrix();
FILE* fd;

int main(int argc, char** argv)
{
    int rank, size;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    printf("Rank %d of %d\n", rank, size);
    numr = rank;
    numt = size;
    
    dt=0.1; t_stop=15.0; t_pause=0.08;
    
    // Выделение массивов
    NM = N*M;
    numBl=numt-1;
    matrixTemp = (double*) malloc(NM*sizeof(double));
    stBl = (int*) malloc((numt+1)*sizeof(int));
    lenBl = (int*) malloc(numt*sizeof(int));
    int* stBl2 = (int*) malloc((numt)*sizeof(int));
    int* lenBl2 = (int*) malloc((numt)*sizeof(int));
    int* lenBlForTemp = (int*) malloc((numt)*sizeof(int));
    
    int i,j,k,rc,p,quot;
    if (!rank)
    {
        // Файл для гнуплота
        fd = fopen("plot","w");
        if(!fd)
        {
            perror("File");
            exit(-1);
        }
        
        // Вычисление начального узла блока и его длины
        for(p=0;p<numt;p++)
        {
            quot = (double)M/numt+0.49;
            lenBl[p] = (p+1==numt?M-p*quot:quot)-(!p?0:1);
            stBl[p] = !p?0:stBl[p-1]+N*lenBl[p-1];
//            printf("lenBL=%d\n",lenBl[p]);
//            printf("stBL=%d\n",stBl[p]);
        }
        stBl[numt] = NM-N*numBl;
        
        // Выделение массивов
        matrixClone = (double*) malloc(NM*NM*sizeof(double));
        matrixCoeff = (double*) malloc(NM*NM*sizeof(double));
        
        gnuplotInit();
        initMatrix();
        //matrixesToFile();
        
    }
    
    // Передача размеров для кластеров
    MPI_Bcast((void *)stBl, numt+1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void *)lenBl, numt, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Выделение массивов для кластеров
    double* a = (double*) malloc(NM*N*lenBl[numr]*sizeof(double));
    double* b = (double*) malloc(N*lenBl[numr]*sizeof(double));

    for(p=0;p<numt;p++)
    {
        lenBl2[p] = NM*N*lenBl[p];
        stBl2[p] = NM*stBl[p];
        lenBlForTemp[p] = N*lenBl[p];
//        printf("lenBL2=%d\n",lenBl2[p]);
//        printf("stBL2=%d\n",stBl2[p]);
//        printf("lenBlForTemp=%d\n",lenBlForTemp[p]);
    }

    // Включение таймера
    if (!rank)
        time_start();
    
    double t;
    int i1,t1;
    for(t=0;t<=t_stop;t+=dt)
    {
        if (!rank)
        {
            memcpy(matrixClone, matrixCoeff, sizeof(double)*NM*NM);
            
            for(i=0;i<N;i++)
            {
                p=0;
                k = stBl[p]+i*lenBl[p];    // Крайний левый узел первого блока
                t1 = lenBl[p];            // Конечный узел первого блока
                for(j=0;j<M;j++)
                {
                    if(j==t1)        // если это окаймление
                    {
                        i1 = NM-1-i*numBl-p++;   // текущий узел в матрице температур
                        k = stBl[p]+i*lenBl[p];     // следующий узел
                        t1 += lenBl[p]+1;            // конечный узел следующего блока
                    }
                    else
                    {
                        i1 = k++;
                    }
                    if(j==0)    // если левая граница
                    {
                        matrixTemp[i1] = values.left_dTdn;
                    }
                    else if(j==M-1)     // если правая граница
                    {
                    }
                    else if(i==N-1)     // если верхняя граница
                    {
                        matrixTemp[i1] = values.top_dTdn;
                    }
                    else if(i==0)       // если нижняя граница
                    {
                    }
                }
            }
            
        }
    
        // Распределение матриц по блокам на прямой метод Гаусса
        MPI_Scatterv(matrixClone, lenBl2, stBl2, MPI_DOUBLE,
                     a, lenBl2[numr], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(matrixTemp, lenBlForTemp, stBl, MPI_DOUBLE,
                     b, lenBlForTemp[numr], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        pryamGauss(a, b);
        
        // Сбор результатов
        MPI_Gatherv(a, lenBl2[numr], MPI_DOUBLE,
                    matrixClone, lenBl2, stBl2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(b, lenBlForTemp[numr], MPI_DOUBLE,
                    matrixTemp, lenBlForTemp, stBl, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Вычисление окаймления
        if (!rank)
            solveMatrix();
        
        // Распределение матриц для обратного метода Гаусса
        MPI_Bcast((void *)matrixTemp, NM, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        obratGauss(a, b);
        
        // Температура в след. момент времени
        MPI_Gatherv(b, lenBlForTemp[numr], MPI_DOUBLE,
                    matrixTemp, lenBlForTemp, stBl, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Рисование в гнуплоте и пауза
        if (!rank)
        {
            drawTemp();
            fprintf(fd,"pause %f\n",t_pause);
        }
    }
    
    if(!rank)
    {
        //matrixesToFile();
        printf("Total Time: %ld msec\n", time_stop());
        fclose(fd);
        free(matrixClone);
        free(matrixCoeff);
    }
    
    free(matrixTemp);
    free(lenBl);
    free(stBl);
    free(stBl2);
    free(lenBl2);
    free(lenBlForTemp);
    
    MPI_Finalize();
    exit(0);
}

void pryamGauss(double* a, double* b)     // Прямой ход Гаусса
{
    int i,j,k,tmp=stBl[numr];
    for(i=0;i<N*lenBl[numr];i++)
    {
        for(k=tmp;k<i+tmp;k++)
        {
            if(!a[NM*i+k])
                continue;
            if(b[i])
                b[i] /= a[NM*i+k];
            if(b[k-tmp])
                b[i] -= b[k-tmp];
            for(j=k+1;j<NM;j++)
            {
                if(j==stBl[numr+1])
                    j=stBl[numt];
                if(a[NM*i+j])
                    a[NM*i+j] /= a[NM*i+k];
                if(a[NM*(k-tmp)+j])
                    a[NM*i+j] -= a[NM*(k-tmp)+j];
            }
            a[NM*i+k] = 0.0;
        }
        
        b[i] /= a[NM*i+(i+tmp)];
        for(j=(i+tmp)+1;j<NM;j++)
        {
            
            if(j==stBl[numr+1])
                j=stBl[numt];
            
            if(a[NM*i+j])
                a[NM*i+j] /= a[NM*i+(i+tmp)];
        }
        a[NM*i+(i+tmp)] = 1.0;
    }
}

void obratGauss(double* a, double* b)     // Обратный ход Гаусса
{
    int i,j,tmp=stBl[numr];
    for(i=N*lenBl[numr]-1;i>=0;i--)
        for(j=i+tmp+1;j<NM;j++)
        {
            if(j==stBl[numr+1])
                j=stBl[numt];
            if(a[NM*i+j])
            {
                if(j<stBl[numt])
                    b[i]-=a[NM*i+j]*b[j-tmp];
                else
                    b[i]-=a[NM*i+j]*matrixTemp[j];
            }
        }
}

void solveMatrix()
{
    int i,j,k;
    
    // Прямой ход Гаусса для окаймления
    for(i=stBl[numt];i<NM;i++)
    {
        for(k=0;k<i;k++)
        {
            if(!matrixClone[NM*i+k])
                continue;
            if(matrixTemp[i])
                matrixTemp[i] /= matrixClone[NM*i+k];
            if(matrixTemp[k])
                matrixTemp[i] -= matrixTemp[k];
            for(j=k+1;j<NM;j++)
            {
                if(matrixClone[NM*i+j])
                    matrixClone[NM*i+j] /= matrixClone[NM*i+k];
                if(matrixClone[NM*k+j])
                    matrixClone[NM*i+j] -= matrixClone[NM*k+j];
            }
            matrixClone[NM*i+k] = 0.0;
        }
        matrixTemp[i] /= matrixClone[NM*i+i];
        for(j=i+1;j<NM;j++)
            if(matrixClone[NM*i+j])
                matrixClone[NM*i+j] /= matrixClone[NM*i+i];
        matrixClone[NM*i+i] = 1.0;
    }
    
    // Обратный ход Гаусса для окаймления
    for(i=NM-1;i>=stBl[numt];i--)
        for(j=i+1;j<NM;j++)
            if(matrixClone[NM*i+j])
                matrixTemp[i]-=matrixClone[NM*i+j]*matrixTemp[j];
}

void initMatrix()   // Заполнение матриц начальными значениями
{
    double deltaX=values.width/(M-1), deltaY=values.height/(N-1);
    //printf("dx=%.2f dy=%.2f\n",deltaX,deltaY);
    int i,j,k,t,p;
    int d,d1,d2,d3,d4,i1,j1;
    
    for(i=0;i<N;i++)
    {
        p=0;
        k = stBl[p]+i*lenBl[p];    // Крайний левый узел первого блока
        t = lenBl[p];            // Конечный узел первого блока
        for(j=0;j<M;j++)
        {
            if(j==t)        // если это окаймление
            {
                i1 = NM-1-i*numBl-p++;   // текущий узел в матрице температур
                d = NM*i1;           // переходим к строке текущего узла в матрице коэффициентов
                d1 = d+k-1;                 // левый соседний узел
                k = stBl[p]+i*lenBl[p];     // следующий узел
                d2 = d+k;                   // правый соседний узел
                d += i1;                    // текущий узел
                d3 = d-numBl;            // верхний соседний узел
                d4 = d+numBl;            // нижний соседний узел
                t += lenBl[p]+1;            // конечный узел следующего блока
            }
            else
            {
                i1 = k++;
                d = NM*i1;
                d1 = d+((j>1 && j==t-lenBl[p])?NM-1-i*numBl-p+1:i1-1);
                d2 = d+((j<M-1 && j==t-1)?NM-1-i*numBl-p:i1+1);
                d += i1;
                d3 = d+lenBl[p];
                d4 = d-lenBl[p];
            }
            
            for(j1=0;j1<NM;j1++)
                matrixCoeff[i1*NM+j1] = 0.0;  // Обнуление матрицы коэффициентов
            
            if(j==0)    // если левая граница
            {
/*                matrixCoeff[d] = 1.0;
                matrixTemp[i1] = 70.0;*/
               matrixCoeff[d2] = 1.0/deltaX;
               matrixCoeff[d] = -1.0/deltaX;
               matrixTemp[i1] = values.left_dTdn;
            }
            else if(j==M-1)     // если правая граница
            {
                matrixCoeff[d] = 1.0;
                matrixTemp[i1] = values.rightTemp;
            }
            else if(i==N-1)     // если верхняя граница
            {
                matrixCoeff[d4] = 1.0/deltaY;
                matrixCoeff[d] = -1.0/deltaY;
                matrixTemp[i1] = values.top_dTdn;
            }
            else if(i==0)       // если нижняя граница
            {
                matrixCoeff[d] = 1.0;
                matrixTemp[i1] = values.bottomTemp;
            }
            else      // если внутренний узел
            {
                matrixCoeff[d] = 1.0+2.0*dt/(deltaX*deltaX)+2.0*dt/(deltaY*deltaY);
                matrixCoeff[d1] = -dt/(deltaX*deltaX);
                matrixCoeff[d2] = -dt/(deltaX*deltaX);
                matrixCoeff[d3] = -dt/(deltaY*deltaY);
                matrixCoeff[d4] = -dt/(deltaY*deltaY);
                matrixTemp[i1] = values.initTemp;
            }
        }
    }
    drawTemp();
    fprintf(fd,"pause 1\n");
}

void drawTemp()     // Рисование в гнуплоте
{
    fprintf(fd,"splot '-' matrix with image\n");
    int i,j,k,t,p;
    for(i=0;i<N;i++)
    {
        p=0;
        k = stBl[p]+i*lenBl[p];
        t = lenBl[p];
        for(j=0;j<M;j++)
            if(j==t)
            {
                fprintf(fd,"%.2f\t",matrixTemp[NM-1-i*numBl-p++]);
                k = stBl[p]+i*lenBl[p];
                t += lenBl[p]+1;
            }
            else
                fprintf(fd,"%.2f\t",matrixTemp[k++]);
        fprintf(fd,"\n");
    }
    fprintf(fd,"e\ne\n");
}

void gnuplotInit()  // Инициализация окна гнуплота
{
    int Tmax=100;
    fprintf(fd,"set title \"MKR: plastina - 8x6cm\"\n");
    fprintf(fd,"unset key\n");
    fprintf(fd,"set tic scale 0\n");
    
    fprintf(fd,"set palette rgbformulae 22,13,-31\n");
    fprintf(fd,"set cbrange [0:%d]\n",Tmax);
    fprintf(fd,"set cblabel \"Temperature [0:%d]\"\n",Tmax);
    fprintf(fd,"unset cbtics\n");
    
    fprintf(fd,"set xrange [0:%d]\n",M-1);
    fprintf(fd,"set yrange [0:%d]\n",N-1);
    
    fprintf(fd,"set view map\n");
}

void matrixesToFile() {
    FILE *f = fopen("out.txt","w");
    int i,j;
    for(i=0;i<NM;i++)
    {
        for(j=0;j<NM;j++)
            fprintf(f,"%.2f\t",matrixCoeff[NM*i+j]);
        fprintf(f,"\t%.2f\n",matrixTemp[i]);
    }
    fclose(f);
}

void time_start()
{
    gettimeofday(&tv1, &tz);
}

long time_stop()
{ 
    gettimeofday(&tv2, &tz);
    dtv.tv_sec=tv2.tv_sec-tv1.tv_sec;
    dtv.tv_usec=tv2.tv_usec-tv1.tv_usec;
    if(dtv.tv_usec<0)
    {
        dtv.tv_sec--;
        dtv.tv_usec+=1000000;
    }
    return dtv.tv_sec*1000+dtv.tv_usec/1000;
}

