import pandas as pd
import numpy as np
import scipy as sc
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import itertools


class Predictor:
    
    def __init__(self, df, ide, pred, real, date = None):
        self.df = df
        self.ide = ide
        self.pred = pred
        self.real = real
        self.date = date
    
    def cortes_ults(self, q_group):
        q = 1- q_group/self.df.shape[0] #el valor pasado siempre debe ser menor que la long del vector
        if q<0:
            print('q_group debe ser menor que ' + str(self.df.shape[0]))
            q=0
        prq = np.quantile(self.df[self.pred], q)
        return prq
    
    
        
    
    
    def performance_table(self, q_group=None):
        l=list(np.arange(0.1, 1, 0.1))
        #l.extend([0.98, 0.99, 0.999])
        l1=list(np.quantile(self.df[self.pred], l))
        
        if q_group != None:
            if isinstance(q_group, list):
                q_groups = np.cumsum(q_group)
                q_groups = map(self.cortes_ults, q_groups)
                l1.extend(q_groups)
            else:
                l1.extend([self.cortes_ults(q_group)])

        l1.extend([0,1])
        l1=list(set(l1))
        l1.sort()
        df_cuts = 1
        df = self.df.copy()
        df.loc[:,self.pred + '_cut'] = pd.cut(df[self.pred], l1, right=True, include_lowest=True)
        df_cuts = df.groupby(self.pred + '_cut').agg({self.ide:'count', self.real:'sum'}).copy()
        
        df_cuts[self.real +'_rate'] = np.round(df_cuts[self.real]/df_cuts[self.ide],3)
        df_cuts['pc_pob'] = np.round(df_cuts[self.ide]/sum(df_cuts[self.ide]),2)
        df_cuts['pc_'+ self.real] = np.round(df_cuts[self.real]/sum(df_cuts[self.real]),2)

        mean= sum(df_cuts[self.real])/sum(df_cuts[self.ide])
        print( 'mean: '+ str(mean ))
        print( 'events: '+ str(sum(df_cuts[self.real])))
        
        df_cuts['lift'] = np.round(df_cuts[self.real +'_rate']/mean,1)
        f1 = []
        pr = []
        re = []
        li = []
        for cut in l1[:-1]:
            tp = sum(df[self.real][df[self.pred]>cut])
            fp = sum(df[self.real][df[self.pred]>cut]==0)
            fn = sum(df[self.real][df[self.pred]<=cut])
            
            #tn = sum(X_.default[X_.pred<=i] == 0)
            _pr = np.round(tp/(tp+fp),2)
            _re = np.round(tp/(tp+fn),2)
            _f1 = np.round(2 * (_pr * _re) / (_pr + _re),2)
            _li = np.round( (tp/(tp+fp))/mean ,1)
            f1.append(_f1)
            pr.append(_pr)
            re.append(_re)
            li.append(_li)
           # print (cut)
        
        df_cuts_2 = df_cuts.copy()
        df_cuts_2['recall_'+ self.real] = re
        df_cuts_2['precision_'+ self.real] = pr
        df_cuts_2['f1_'+ self.real] = f1
        df_cuts_2['lift_'+ self.real] = li
        quantities = list(df_cuts_2[self.ide])
        quantities.reverse()
        quantities = list(np.cumsum(quantities))
        quantities.reverse()
        df_cuts_2['cum_' +str(self.ide)] = quantities
        
        
        df_cuts_2 = df_cuts_2[['cum_' +str(self.ide), 'precision_'+ self.real, \
                               'recall_'+ self.real,'f1_'+ self.real ,\
                              'lift_'+ self.real
                              ]]
  
        df_cuts_2.index = l1[:-1]
        df_cuts_2.index.name = '> cut'
        return (df_cuts , df_cuts_2)
    
    
    
    
    


    def graph_ks(self):
        print(stats.ks_2samp(self.df[self.pred][self.df[self.real]==0], self.df[self.pred][self.df[self.real]==1]))
        tbla_1 = self.performance_table()[0]
        tbla_1['no_target']=tbla_1[self.ide]-tbla_1[self.real]
        
        tbla_1['pc_no_target']=tbla_1['no_target'].cumsum()/sum(tbla_1['no_target'])
        tbla_1['pc_target']=tbla_1[self.real].cumsum()/sum(tbla_1[self.real])
        tbla_1['pc_todos']=tbla_1[self.ide].cumsum()/sum(tbla_1[self.ide])

        fig, ax = plt.subplots()
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        pc_target=list(tbla_1.pc_target)
        pc_target.insert(0,0)
        pc_no_target=list(tbla_1.pc_no_target)
        pc_no_target.insert(0,0)
        pc_todos=list(tbla_1.pc_todos)
        pc_todos.insert(0,0)

        plt.plot(pc_todos, pc_target , color='r', marker='d', label='% cum ' + str(self.real))
        plt.plot(pc_todos, pc_no_target , color='k', marker='d',label='% cum not '+ str(self.real))
        plt.xlabel('% cumulative total population sorted by proba: '+ str(self.pred))
        plt.ylabel('% cumulative rate ' +str(self.real) + ' & not ' + str(self.real))
        plt.title('Performance score')
        plt.grid()
        legend = ax.legend(loc='upper left', shadow=False, fontsize='large')
        return None

 
    def graph_others(self):

        tbla_1 = self.performance_table()[1]
       
        
        
        fig, ax = plt.subplots()
        plt.ylim(-0.05, 1.05)
        plt.xlim(-0.05, 1.05)
        _re = 0
        #_pr = sum(self.df[self.real])/self.df.shape[0]
        
        re = list(tbla_1['recall_'+ self.real] )
        re.append(_re)
        
        pr = list(tbla_1['precision_'+ self.real])
        #pr.append(_pr)
        
        f1 = list(tbla_1['f1_'+ self.real])
        #f1.append( np.round(2 * (_pr * _re) / (_pr + _re),2))
        
        li = list(tbla_1['lift_'+ self.real])
        #li.append(1) 
            
        #print(pr)
        x = np.arange(0,1.1,0.1)
        ax.plot(x[:-1], pr , color='r', marker='d', label='% precision ' + str(self.real))
        ax.plot(x, re , color='b', marker='d',label='% recall '+ str(self.real))
        ax.plot(x[:-1], f1 , color='k', marker='d',label='% f1 '+ str(self.real))
        ax.set_xlabel('% population')
        ax.set_ylabel('metrics')
        ax.set_title('Performance metrics')
        plt.grid()
       

        ax2=ax.twinx()
        ax2.plot(x[:-1],li, color='g', marker='d',label='% lift')
        ax2.set_ylabel('lift', color = 'g')
        
        legend = ax.legend(loc='center right', shadow=False, fontsize='large', bbox_to_anchor=(-0.08, 0.85))
        #legend2 = ax2.legend(loc=0, shadow=False, fontsize='large')
        return None

    
    
    
    def metricas_performance(self, date = None):
        target = self.df[self.real]
        prediction = self.df[self.pred]
        
        if date != None:
            index = self.df[self.date] == date
            target = target[index]
            prediction= prediction[index]
        
        ceros = (prediction[(target.values == 0)])
        unos = (prediction[(target.values == 1)])
        auc = round(roc_auc_score(target,prediction),2)
        ks = round(ks_2samp(ceros, unos)[0],4)
        br = sum(target)/len(target)
        #print('auc: '+ str(auc)+' - ks: '+str(ks))
        return {'auc':auc, 'ks':ks, self.real+'_pc':  br}
    
    
    
    def stability(self):
        ds = list(set(self.df[self.date]))
        ds.sort()
        stability = []
        for d in ds:
            data = { self.date: d }
            data.update(self.metricas_performance(d))
            stability.append(data)
        st_df = pd.DataFrame(stability)
        
    
        return st_df
    
    
    def graph_stability(self):
        a= self.stability()
        
        fig, ax = plt.subplots()
        # make a plot
        lns1 = ax.plot(a.index, a.auc, color="red", marker="o",  label='auc')
        lns2 = ax.plot(a.index, a.ks, color="blue", marker="o", label='ks')
        ax.set_xlabel("date",fontsize=14)
        ax.set_ylabel("performance",color="black",fontsize=14)
        plt.xticks(rotation=90)

        ax2=ax.twinx()
        
        lns3= ax2.plot(a.index, a[self.real+'_pc'], color="black", marker="o", label=self.real+'_pc')
        ax2.set_ylabel(self.real+'_pc', color="black",fontsize=14)
        lns = lns1+lns2#+lns3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='center left', bbox_to_anchor=(-0.35, 0.5))
        lns = lns3
        labs = [l.get_label() for l in lns]
        ax2.legend(lns3, labs, loc='center left', bbox_to_anchor=(1.2, 0.5))
        
        plt.xticks(a.index, list(map(lambda x: x.strftime('%Y-%m-%d'), a[self.date])))
        plt.title('performance stability')
        plt.show()

        
        

class Events:
    def __init__(self, df, grouper, datetime_name, event_name, every_x_minutes=30):
        self.grouper = grouper
        self.datetime_name = datetime_name
        self.event_name = event_name
        self.every_x_minutes = every_x_minutes
        self.__fill_state_of_events(df)

    def __fill_state_of_events(self, df):
        if self.every_x_minutes>60:
            print('minutes should be <= 60')
            self.every_x_minutes=60
        #tengo el problema de que si es de a 2 horas no sabe si son las pares o impares     
        df = df.drop_duplicates() 
        df = df[df[self.event_name].isnull()==False]
        unidad = str(self.every_x_minutes)+'min'
        new_name = self.event_name+'_'+unidad
        df[new_name] = df[self.datetime_name].dt.floor(unidad)
        df = df.sort_values(by=[self.grouper, self.datetime_name]) #importante para el paso de quedarse con le ultimo por hora
        df = df.groupby([self.grouper, new_name], as_index= False, sort=False ).nth([-1]) 

        max_fecha=max(df[self.datetime_name])
        #print(max_fecha)
        fechas_l=pd.date_range(start=min(df[self.datetime_name]), end=max_fecha,  freq=unidad)
        #print(fechas_l)
        instals=list(set(df[self.grouper]))


        fechas_df=pd.DataFrame(list(itertools.product(fechas_l, instals)))
        fechas_df.columns=[new_name , self.grouper]

        res=pd.merge(fechas_df, 
                     df[[ self.grouper, new_name, self.event_name]], 
                     left_on=[new_name, self.grouper],
                     right_on=[new_name, self.grouper],
                     how='left')

        res=res.sort_values(by=[self.grouper, new_name]) #importante para el paso de llenar con el anterior
        #lleno con el estado anterior
        for i in [self.grouper, self.event_name]:
            res[i+str('_1')]=res[i].fillna(method='ffill')
            res=res.drop(i,axis=1) 
        res.columns=[new_name, self.grouper, self.event_name]
        res.index=np.arange(0, res.shape[0], 1)
        self.event_with_period_name = new_name
        res=res[res[self.event_name].isnull()==False]
        self.df = res
        return None
        
    def plot_ide(self, ide_value, event_value_pos, figsize=(10,5)):
        
        a = self.df[self.df[self.grouper]== ide_value ].copy()
        
        new_name='f_'+self.event_name
        
        a[new_name]=0
        a.loc[a[self.event_name] == event_value_pos, new_name]=1
        
        #falta q agrupe por semana y haga varias series por semana
        
        a['dow_name']=a[self.event_with_period_name].dt.day_name()
        a['dow']=a[self.event_with_period_name].dt.dayofweek
        a['week']=a[self.event_with_period_name].dt.isocalendar().week 
        
        a['week_dow']= (a['week']).astype(str) + (a['dow']).astype(str)
        
        a['time']=list(map(lambda x: x.strftime('%H-%M'), a[self.event_with_period_name ]))
        #print(a)
        days_week =list(set(a['dow']))
        days_week.sort()
        
        #print(days_week)
        
        l= list(set(map(lambda x: x.strftime('%H-%M'), a[self.event_with_period_name])))
        l.sort()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        symbols = ["o", "v", "x",  "s", "p", "P", "+", ".", ",", "X", "D", "d", "^", "<", ">"]
        j=0
        #print(days_week)
        for i in days_week:
            b = a.copy()
            b=b[b[self.event_name].isnull()==False]
            b=b[b.dow==i]
            b=b.sort_values('time')
            b=b.set_index(np.arange(0, b.shape[0],1))
            fig, ax = plt.subplots(figsize=figsize)
            #print(b)
            dow_name=b['dow_name'][0]
            plt.title('state for ide '+ str(ide_value)+'_ weekday: '+ str(dow_name))
            
            l= list(set(map(lambda x: x.strftime('%H-%M'), a[self.event_with_period_name ])))
            l.sort()
            plt.xticks(rotation=90)
            ax.set_ylim(-0.02,1.02)
            #plt.xticks(l, l)   
            ax.set_xlabel("timestamp", fontsize=14)
            ax.set_ylabel(self.event_with_period_name , color="black", fontsize=14)

            plt.plot(b['time'], b[new_name], f'{colors[j % len(colors)]}{symbols[int(j / len(colors)) % len(symbols)]}-', label= i )
            j=j+1
            plt.show()
        
        return None

