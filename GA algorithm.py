import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(seed=0)

def bin_to_dec(X):
    x=0
    y=0
    for i in range(nbit):
       x = x + X[i]*2**i 
       y = y + X[i+8]*2**i
    x = 8+ x * 2 / (2**8-1)
    y = 10+ y* 3 / (2**8-1)
    return x,y
def fun(X):
    
    x,y=bin_to_dec(X)

    ans = -x*np.sin(4*x) - 1.1*y*np.sin(2*y)+1
    return x,y,ans

D = 2
maxit = 3000
population_size = 16

nbit = 8
mr = 0.2
cr = 0.9
maxlog = []
num1cool = []
num2cool = []

chromosome = D * nbit
max_solution = 0
pos_solution = [0,0]

population = np.ones((population_size,chromosome))
for i in range(population_size):
    for j in range(chromosome):
        population[i,j]=np.random.randint(0,2)
    x,y=bin_to_dec(population[i])
    while x+y>22:
        for j in range(chromosome):
            population[i,j]=np.random.randint(0,2)
            x,y=bin_to_dec(population[i])

decx,decy,value = fun(population)

def select(X):
    arr = np.argsort(X)
    percentage = []
    cumulative = []
    for i in range(population_size):
        percentage.append(1/((population_size+1)*population_size/2)*(i+1))

    valuepercentage = np.zeros(population_size)
    j=0
    for i in arr:
        valuepercentage[i] =percentage[j]
        j+=1
    
    cumulative = np.cumsum(valuepercentage,dtype = float)
       
    selection = []
    x=np.random.uniform(0,1)

    k=0
    while True:
        
        if x<cumulative[k]:
            break
        k=k+1
        if k==population_size:
            k=0
            break
   
    selection.append(X[k])
    value_new = np.delete(X,k)
    
    arr_new = np.argsort(value_new)
    percentage_new = []
    cumulative_new = []
    for i in range(population_size-1):
        percentage_new.append(1/((population_size+1-1)*(population_size-1)/2)*(i+1))
        
    valuepercentage_new = np.zeros(population_size-1)
    j=0
    for i in arr_new:
        valuepercentage_new[i]=percentage_new[j]
        j+=1
    
    cumulative_new = np.cumsum(valuepercentage_new,dtype = float)
    
    x=np.random.uniform(0,1)

    k=0
    while True:
        if x<cumulative_new[k]:
            break
        k+=1
        if k==population_size-1:
            k=0
            break
    selection.append(value_new[k])
    num1 = np.where( X==selection[0])
    num2 = np.where( X==selection[1])
    
    return selection,num1,num2

def crossover(X,Y):
    
    docr = False
    ax = population[X]   
    bx = population[Y]     
    if np.random.rand(1) <= cr:
        a = population[X]
        b = population[Y]
        x1,y1 = bin_to_dec(a[0])  
        x2,y2 = bin_to_dec(b[0])

        err=0
        while (docr == False or x1+y1>22 or x2+y2>22):    
            ax=[]
            ay=[]
            bx=[]
            by=[]
            for i in range(nbit):
                ax.append(a[0][i])
                bx.append(b[0][i])
                ay.append(a[0][i+nbit])
                by.append(b[0][i+nbit])
            t = math.floor(np.random.rand(1)*(nbit))
            temp = np.zeros(nbit)
            temp[0:t],ax[0:t],bx[0:t]=ax[0:t],bx[0:t],ax[0:t]
            t = math.floor(np.random.rand(1)*(nbit))
            temp[0:t],ay[0:t],by[0:t]=ay[0:t],by[0:t],ay[0:t]
            for i in ay :
                ax.append(i)
                        
            for i in by:
                bx.append(i)

            x1,y1 = bin_to_dec(ax)  
            x2,y2 = bin_to_dec(bx)  
            docr = True
            err+=1
            if(x1+y1>22 or x2+y2>22):

                    while(err==5 or x1+y1>22 or x2+y2>22):
                        ax=[]
                        bx=[]
                        for i in range(population_size):
                            ax.append(np.random.randint(0,2))
                            bx.append(np.random.randint(0,2))
                        x1,y1 = bin_to_dec(ax)  
                        x2,y2 = bin_to_dec(bx)
                        err+=1

                    break
                        
                    
            
    return ax,bx,docr


def mutation(bool1,dad,mom):
    
    dad = dad
    mom = mom
    x1 =100
    y1 =100

    if bool1 == True:
        a=np.random.rand(1)
        b=np.random.rand(1)


        if a<= mr:
            while(x1+y1>22):
                x = np.floor(np.random.rand(1)*(nbit*D))
                dad[int(x[0])]= abs(dad[int(x[0])]-1)
                x1,y1=bin_to_dec(dad)
                bool1=False
        bool1 = True
        if b<= mr:
            while( bool1==True or x1+y1>22):
                x = np.floor(np.random.rand(1)*(nbit*D))
                mom[int(x[0])]= abs(mom[int(x[0])]-1)
                x1,y1=bin_to_dec(mom)
                bool1=False
    return dad,mom

selection,num1_arg,num2_arg = select(value)

aco1,aco2,docr1 = crossover(num1_arg,num2_arg)
child1,child2 = mutation(docr1,aco1,aco2)

it = 1
maxmax = []

while (it<=maxit):

    num1cool.append(num1_arg)
    num2cool.append(num2_arg)
    
    population[num1_arg]=child1
    population[num2_arg]=child2
    
    decx,decy,value = fun(population)
    
    maxlog.append(np.max(value))
    
    if np.max(value) >= max_solution:
        max_solution = np.max(value)
        pos_solution[0] = decx[np.where(value==np.max(value))]
        pos_solution[1] = decy[np.where(value==np.max(value))]
    selection,num1_arg,num2_arg = select(value)
    aco1,aco2,docr1 = crossover(num1_arg,num2_arg)
    child1,child2 = mutation(docr1,aco1,aco2)
    
    

    print("第",it,"個 : ", np.max(maxlog),"位置 :",pos_solution[0][0],pos_solution[1][0],"變動值 :",value[num1cool[it-1]]," ",value[num2cool[it-1]])
    
    maxmax.append(np.max(maxlog))
   
    it+=1
'''
print("maximum solution: ",max_solution,"position: ",pos_solution[0][0],pos_solution[1][0])

plt.xlabel("Generation")
plt.ylabel("Value")
plt.plot(np.arange(1,len(maxmax)+1),maxmax,label='max value')
plt.legend(loc='lower right')
plt.show()

plt.xlabel("Generation")
plt.ylabel("Value")
plt.rcParams["figure.figsize"] = (14, 4)
plt.plot(np.arange(1,len(maxmax)+1),maxmax,label='max value')
plt.plot(np.arange(1,len(maxmax)+1),maxlog,'--',label='maxlog value')

plt.legend(loc='lower right')
plt.show()

xx=np.arange(8,10,0.03)
xx=np.arange(8,10,0.02)
'''