import numpy as np 

#%%

def continuous_angle(x):
    
    last = 0
    out = []

    for angle in x:
        while angle < last-np.pi: angle += 2*np.pi
        while angle > last+np.pi: angle -= 2*np.pi
        last = angle
        out.append(angle)

    return np.array(out)

#%%

def dist2agent(data): 
    
    for i in range(1,10):
        
        x = data[' x%d'%i]
        y = data[' y%d'%i]
        xa = data[' x0']
        ya = data[' y0']
        data[' dist%d'%i] = np.sqrt((x-xa)**2+(y-ya)**2)
    return data



def poly2(data):

    for i in range(10):
        
        
        x = data[' x%d' % i]
        y = data[' y%d' % i]
        
        data[' x2%d' % i] = x**2
        data[' y2%d' % i] = y**2
        data[' xy%d' % i] = x*y
        
    return data
        

def speed_direction(data):

    for i in range(10):
        
        speed = np.zeros(11)
        sin_dir = np.zeros(11)
        cos_dir = np.zeros(11)
        
        x = data[' x%d' % i]
        y = data[' y%d' % i]
        
        speed[0] = np.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2)
        direction = np.arctan2(y[1]-y[0],x[1]-x[0])
        sin_dir[0] = np.sin(direction)
        cos_dir[0] = np.cos(direction)
        
        speed[10] = np.sqrt((x[10]-x[9])**2+(y[10]-y[9])**2)
        direction = np.arctan2(y[10]-y[9],x[10]-x[9])
        sin_dir[10] = np.sin(direction)
        cos_dir[10] = np.cos(direction)
        
        for t in range(1,10):
            
            speed[t] = np.sqrt((x[t+1]-x[t-1])**2+(y[t+1]-y[t-1])**2)/2
            direction = np.arctan2(y[t+1]-y[t-1],x[t+1]-x[t-1])
            sin_dir[t] = np.sin(direction)
            cos_dir[t] = np.cos(direction)
            
            
        data[' speed%d' % i] = speed
        data[' sin(dir)%d' % i] = sin_dir
        data[' cos(dir)%d' % i] = cos_dir
        
    return data



def acceleration(data):
    
    for i in range(10):
        
        a = np.zeros(11)
        
        speed = data[' speed%d' % i]
        
        a[0] = speed[1]-speed[0]
        a[10] = speed[10]-speed[9]
        
        for t in range(1,10):
            a[t] = (speed[t+1]-speed[t-1])/2
            
            
        data[' acceleration%d' % i] = a
        
    return data




def turning(data):
    
    for i in range(10):
        
        turn = np.zeros(11)
        
        sin_dir = data[' sin(dir)%d' % i]
        cos_dir = data[' cos(dir)%d' % i]
        direction = np.arctan2(sin_dir, cos_dir)
        direction = continuous_angle(direction)
        
        turn[0] = direction[1]-direction[0]
        turn[10] = direction[10]-direction[9]
        
        for t in range(1,10):
            turn[t] = (direction[t+1]-direction[t-1])/2
            
            
        data[' turning%d' % i] = turn
        
    return data
    

def replace_agent(data):
    for i in range(10):
        if data[' role%d' % i][0] == ' agent':
            temp = data[[' id0',' role0',' type0',' x0',' y0',' present0']]
            data[[' id0',' role0',' type0',' x0',' y0',' present0']] = data[[' id%d'%i,' role%d'%i,' type%d'%i,' x%d'%i,' y%d'%i,' present%d'%i]]
            data[[' id%d'%i,' role%d'%i,' type%d'%i,' x%d'%i,' y%d'%i,' present%d'%i]] = temp
            
            
    return data
            
            
def empty_fix(data, x_max=30, y_max=10):
    xs = np.array([2*x_max,2*x_max,-2*x_max,-2*x_max,3*x_max,3*x_max,-3*x_max,-3*x_max,4*x_max])
    ys = np.array([2*y_max,-2*y_max,2*y_max,-2*y_max,3*y_max,-3*y_max,3*y_max,-3*y_max,-4*y_max])
    j = 0
    for i in range(1,10):
        if data[' present%d'%i][0] == 0:
            data[' x%d'%i] = xs[j]*np.ones(11)
            data[' y%d'%i] = ys[j]*np.ones(11)
            j += 1
            data[' role%d'%i] = ' others'
            data[' type%d'%i] = ' car'
    return data

