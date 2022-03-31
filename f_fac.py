# Determinación de los factores de corrección de la simulación
# Por Sebastián Sarasti
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from d_funciones import netcount, spectrum, normalizar_exp, area
# se cargan los espectros del Cs
cs_e={'5cm':np.loadtxt('DetectorNaI30mar21-Cs137-5cm-45min.TKA'),
      '10cm':np.loadtxt('DetectorNaI30mar21-Cs137-10cm-109min.TKA'),
      '15cm':np.loadtxt('DetectorNaI31mar21-Cs137-15cm-217min.TKA')}
#CREACION DE LOS CANALES Y LAS ENERGIAS
canal=np.linspace(1,8190,8190)
Energy=np.array([((i*0.2157-11.88)*0.001) for i in canal])
# se extraen los tiempos de datacion
t_cs_e={"5cm":0, "10cm":0,"15cm":0}
for i in t_cs_e:
    t_cs_e[i]=cs_e[i][1]
t_cs_e
# se determina el conteo neto
cs_e_n={'5cm':[],'10cm':[], '15cm':[]}
bk=np.loadtxt("DetectorNaI21oct20fondo2hFuer.txt")
for i in cs_e_n:
    cs_e_n[i]=netcount(cs_e[i],bk)
# se cargan los datos simulados
cs_s={'5cm':np.loadtxt('./NaI_new/'+'NaIDetector_5cm_cs137.dat'),
      '10cm':np.loadtxt('./NaI_new/'+'NaIDetector_10cm_cs137.dat'),
      '15cm':np.loadtxt('./NaI_new/'+'NaIDetector_15cm_cs137.dat')}
# se realizan los histogramas para cada espectro simulado
for i in cs_s:
    cs_s[i]=np.histogram(cs_s[i],bins=1500)
# se agrega la resolucion del sistema
cs_s_r={}
cs_s_r['5cm']=spectrum(cs_s['5cm'][1], cs_s['5cm'][0], 0.026)
cs_s_r['10cm']=spectrum(cs_s['10cm'][1], cs_s['10cm'][0], 0.026)
cs_s_r['15cm']=spectrum(cs_s['15cm'][1], cs_s['15cm'][0], 0.026)
t_simulacion = lambda p, actividad: p/actividad
cs_s_r1={'5cm':cs_s_r['5cm'][1] , '10cm':cs_s_r['10cm'][1] , '15cm':cs_s_r['15cm'][1]}
# NORMALIZACION EN FUNCION DE LA ACTIVIDAD DE LA FUENTE
# Se cargan las actividades
actividades=pd.read_csv('actividad.csv')
# se agrega la actividad al espectro simulado
for i in cs_s_r1:
    cs_s_r1[i]=cs_s_r1[i]/t_simulacion(1e6,actividades.iloc[0,1])
# Se eliminan los rayos X del espectro experimental
def without_x_ray(spectra, energia):
    for i in range(len(spectra)):
        if energia[i]<0.04:
            spectra[i]=0
        else:
            spectra[i]=spectra[i]
    return spectra
for i in cs_e_n:
    cs_e_n[i]=without_x_ray(cs_e_n[i], Energy)
# SE CREA UNA FUNCION QUE DETERMINAR EL FACTOR DE CORRECCION  
def f_correcion(d_spectra_e, d_spectra_s, activity, p_simuladas, nuclei):
    """
    This function is able to estimate and give spectrum with activity 
    correction and help to estimate correction factor due to PTM and 
    scintillation procces related
    Where:
        d_spectra_e: dictionary with net count rate spectrum
        d_spectra_e: dictionary with simulated spectrum with resolution 
        and activity scaling
        activity: sample activity at sampling day
        p_simuladas: number of particles in MC
    """
    if nuclei=='Eu':
        # se determina el máximo de cada espectro experimental
        m_e_=[]
        for i in d_spectra_e:
            m_e_.append(max(d_spectra_e[i][500:2000]))
        # se determina el máximo de cada espectro simulado
        m_s_=[]
        for i in d_spectra_s:
            m_s_.append(max(d_spectra_s[i][70:1500]))
        #se determina el factor de corrección en función del fotopico máximo de cada
        # espectro
        f_pmt_=np.array([m_s_[i]/m_e_[i] for i in range(len(m_s_))])
    else:
        # se determina el máximo de cada espectro experimental
        m_e_=[]
        for i in d_spectra_e:
            m_e_.append(max(d_spectra_e[i]))
        # se determina el máximo de cada espectro simulado
        m_s_=[]
        for i in d_spectra_s:
            m_s_.append(max(d_spectra_s[i]))
        #se determina el factor de corrección en función del fotopico máximo de cada
        # espectro
        f_pmt_=np.array([m_s_[i]/m_e_[i] for i in range(len(m_s_))])
    return f_pmt_ 
b_cs=f_correcion(cs_e_n, cs_s_r1, actividades.iloc[0,1], 1e6,'Cs')
b_cs=[b_cs.mean(),np.std(b_cs)]
fig1=plt.figure(figsize=(5,5))
ax1=plt.axes()
plt.plot(Energy,cs_e_n['15cm'],'-r',label="5cm experimental")
plt.plot(cs_s_r['15cm'][0],cs_s_r1['15cm'], label="5cm simulated")
plt.plot(Energy,cs_e_n['10cm'],'-b', label="10cm experimental")
plt.plot(Energy,cs_e_n['15cm'],'-g',label="15cm experimental")
ax1.plot(cs_s_r['5cm'][0],cs_s_r1['5cm'],':r', label="5cm simulated")
ax1.plot(cs_s_r['10cm'][0],cs_s_r1['10cm']/b_cs,':b',label="10cm simulated")
ax1.plot(cs_s_r['15cm'][0],cs_s_r1['15cm']/b_cs,':g',label="15cm simulated")
plt.legend()
plt.grid()
#plt.xlim(0.5, 1)
#plt.title('Photopeak of Cs-137')
co_e={'5cm':np.loadtxt('DetectorNaI30mar21-Co60-5cm-31min.TKA'),
      '10cm':np.loadtxt('DetectorNaI30mar21Co-60-10cm-72min.TKA'),
      '15cm':np.loadtxt('DetectorNaI31mar21-Co-60-15cm-141min.TKA')}  
# se determina la tasa de conteo neto
co_e_n={'5cm':[],'10cm':[], '15cm':[]}
for i in co_e_n:
    co_e_n[i]=netcount(co_e[i],bk)
# se carga la simulacion
co_s={'5cm':np.loadtxt('NaIDetector_5cm_co60.dat'),
      '10cm':np.loadtxt('NaIDetector_10cm_co60.dat'),
      '15cm':np.loadtxt('NaIDetector_15cm_co60.dat')} 
# se realizan los histogramas para cada espectro simulado
for i in co_s:
    co_s[i]=np.histogram(co_s[i],bins=1500)
# se agrega la resolucion del sistema
co_s_r={}
co_s_r['5cm']=spectrum(co_s['5cm'][1], co_s['5cm'][0], 0.032)
co_s_r['10cm']=spectrum(co_s['10cm'][1], co_s['10cm'][0], 0.032)
co_s_r['15cm']=spectrum(co_s['15cm'][1], co_s['15cm'][0], 0.032)
co_s_r1={'5cm':co_s_r['5cm'][1] , '10cm':co_s_r['10cm'][1] , '15cm':co_s_r['15cm'][1]}
# NORMALIZACION EN FUNCION DE LA ACTIVIDAD DE LA FUENTE
# se agrega la actividad al espectro simulado
for i in co_s_r1:
    co_s_r1[i]=co_s_r1[i]/t_simulacion(1e7,actividades.iloc[2,1])
b_co=f_correcion(co_e_n, co_s_r1, actividades.iloc[2,1], 1e7,'Co')
b_co=[b_co.mean(),np.std(b_co)]
#fig1=plt.figure(figsize=(5,5))
#ax1=plt.axes()
#plt.plot(Energy,co_e_n['5cm'],'-r',label="5cm experimental")
#plt.plot(Energy,co_e_n['10cm'],'-b', label="10cm experimental")
#plt.plot(Energy,co_e_n['15cm'],'-g',label="15cm experimental")
#ax1.plot(cs_s_r['5cm'][0],co_s_r1['5cm']/b_co,':r', label="5cm simulated")
#ax1.plot(cs_s_r['10cm'][0],co_s_r1['10cm']/b_co,':b',label="10cm simulated")
#ax1.plot(cs_s_r['15cm'][0],co_s_r1['15cm']/b_co,':g',label="15cm simulated")
#plt.legend()
#plt.grid()
#plt.title('Photopeak of Cs-137')
# se cargan los datos del eu-152
eu_e={'5cm':np.loadtxt('DetectorNaI30mar21Eu152-5cm-7min.TKA'),
      '10cm':np.loadtxt('DetectorNaI30mar21Eu152-10cm-18min.TKA'),
      '15cm':np.loadtxt('DetectorNaI31mar21-Eu152-15cm-39min.TKA')}  
# se determina la tasa de conteo neto
eu_e_n={'5cm':[],'10cm':[], '15cm':[]}
for i in eu_e_n:
    eu_e_n[i]=netcount(eu_e[i],bk)
# se carga la simulacion
eu_s={'5cm':np.loadtxt('NaIDetector_5cm_Eu152.dat'),
      '10cm':np.loadtxt('NaIDetector_10cm_Eu152.dat'),
      '15cm':np.loadtxt('NaIDetector_15cm_Eu152.dat')}
# se realizan los histogramas
for i in co_s:
    eu_s[i]=np.histogram(eu_s[i],bins=500)
# se agrega la resolucion del sistema
eu_s_r={}
eu_s_r['5cm']=spectrum(eu_s['5cm'][1], eu_s['5cm'][0], 0.02)
eu_s_r['10cm']=spectrum(eu_s['10cm'][1], eu_s['10cm'][0], 0.02)
eu_s_r['15cm']=spectrum(eu_s['15cm'][1], eu_s['15cm'][0], 0.02)
eu_s_r1={'5cm':eu_s_r['5cm'][1] , '10cm':eu_s_r['10cm'][1] , '15cm':eu_s_r['15cm'][1]}
# NORMALIZACION EN FUNCION DE LA ACTIVIDAD DE LA FUENTE
# se agrega la actividad al espectro simulado
for i in eu_s_r1:
    eu_s_r1[i]=eu_s_r1[i]/t_simulacion(1e6,actividades.iloc[1,1]*0.2831)
b_eu=f_correcion(eu_e_n, eu_s_r1, actividades.iloc[1,1]*0.2831, 1e6, "Eu")
b_eu=[b_eu.mean(),np.std(b_eu)]
fig1=plt.figure(figsize=(5,5))
ax1=plt.axes()
plt.plot(Energy,eu_e_n['5cm'],'-r',label="5cm experimental")
#plt.plot(Energy,eu_e_n['10cm'],'-b', label="10cm experimental")
#plt.plot(Energy,eu_e_n['15cm'],'-g',label="15cm experimental")
ax1.plot(cs_s_r['5cm'][0],eu_s_r1['5cm']/b_eu[0],'-g', label="5cm simulated")
plt.xlim(0.2,0.4)
plt.grid()
#ax1.plot(cs_s_r['10cm'][0],eu_s_r1['10cm']/b_eu[0],':b',label="10cm simulated")
#ax1.plot(cs_s_r['15cm'][0],eu_s_r1['15cm']/b_eu[0],':g',label="15cm simulated")
plt.legend()
plt.grid()
plt.title('Photopeak of Eu-152')
# Se cargan los datos del Am
am_e={'5cm':np.loadtxt('DetectorNaI30mar21-Am241-5cm-8min.TKA'),
      '10cm':np.loadtxt('DetectorNaI30mar21-Am241-10cm-23min.TKA'),
      '15cm':np.loadtxt('DetectorNaI31mar21-Am241-15cm-51min.TKA')}  
# se determina la tasa de conteo neto
am_e_n={'5cm':[],'10cm':[], '15cm':[]}
for i in eu_e_n:
    am_e_n[i]=netcount(am_e[i],bk)
# se cargan los datos simulados
am_s={'5cm':np.loadtxt('NaIDetector_5cm_Am241.dat'),
      '10cm':np.loadtxt('NaIDetector_10cm_Am241.dat'),
      '15cm':np.loadtxt('NaIDetector_15cm_Am241.dat')}
for i in am_s:
    am_s[i]=np.histogram(am_s[i],bins=500)
# se agrega la resolucion del sistema
am_s_r={}
dist_l=['5cm','10cm','15cm']
for distancia in dist_l:
    am_s_r[distancia]=spectrum(am_s[distancia][1], am_s[distancia][0], 0.005)
am_s_r1={'5cm':am_s_r['5cm'][1] , '10cm':am_s_r['10cm'][1] , '15cm':am_s_r['15cm'][1]}
# se agrega la actividad al espectro simulado
for i in am_s_r1:
    am_s_r1[i]=am_s_r1[i]/t_simulacion(1e6,actividades.iloc[3,1]*0.357)
b_am=f_correcion(am_e_n, am_s_r1, actividades.iloc[3,1]*0.357, 1e6, "Am")
b_am=[b_am.mean(),np.std(b_am)]
fig5=plt.figure(figsize=(5,5))
plt.plot(am_s_r['5cm'][0],am_s_r1['5cm'],'g',label='5cm')
plt.plot(am_s_r['10cm'][0],am_s_r1['10cm'],'r',label='10cm')
plt.plot(am_s_r['15cm'][0],am_s_r1['15cm'],'b',label='15cm')
plt.legend()
#eu_s_r['5cm']=spectrum(eu_s['5cm'][1], eu_s['5cm'][0], 0.01)
#eu_s_r['10cm']=spectrum(eu_s['10cm'][1], eu_s['10cm'][0], 0.01)
#eu_s_r['15cm']=spectrum(eu_s['15cm'][1], eu_s['15cm'][0], 0.01)
#eu_s_r1={'5cm':eu_s_r['5cm'][1] , '10cm':eu_s_r['10cm'][1] , '15cm':eu_s_r['15cm'][1]}
# crear un dataframe que almacene los datos
factores_correccion=pd.DataFrame(columns=['Factor','Desviación estándar'],index=['Am-241','Eu-152','Cs-137','Co-60'])
en_=np.array([59.5,120,662,1170])
factores_correccion.iloc[0,:]=b_am
factores_correccion.iloc[1,:]=b_eu
factores_correccion.iloc[2,:]=b_cs
factores_correccion.iloc[3,:]=b_co
factores_correccion['Energia (keV)']=en_
# grafica de la relación entre los factores de corrección y la energía más el error
x=factores_correccion['Energia (keV)']
y=factores_correccion['Factor']
error=factores_correccion['Desviación estándar']
fig=plt.figure(dpi=2000)
plt.errorbar(x, y, yerr=error,ecolor='r',capsize=10)
plt.grid()
plt.ylabel('Correction factor $(adimensional)$')
plt.xlabel('Energy (keV)')
plt.savefig("f_correction.jpg")
# se determina el FEPE simulado
a=['5cm','10cm','15cm']
areas=np.zeros((3,7))
A=[]
for i in a:
    A.append(area(eu_s_r1[i],eu_s_r[i][0],75,105)/b_eu[0])
areas[:,1]=A
A=[]
for i in a:
    A.append(area(eu_s_r1[i],eu_s_r[i][0],166,195)/b_eu[0])
areas[:,2]=A
A=[]
for i in a:
    A.append(area(eu_s_r1[i],eu_s_r[i][0],240,276)/b_eu[0])
areas[:,3]=A
A=[]
for i in a:
    A.append(area(cs_s_r1[i],cs_s_r[i][0],456,533)/b_cs[0])
areas[:,4]=A
A=[]
for i in a:
    A.append(area(co_s_r1[i],co_s_r[i][0],825,918)/b_co[0])
areas[:,5]=A
A=[]
for i in a:
    A.append(area(co_s_r1[i],co_s_r[i][0],955,1064)/b_co[0])
areas[:,6]=A
A=[]
for i in a:
    A.append(area(am_s_r1[i],am_s_r[i][0],31,54)/b_am[0])
areas[:,0]=A
# DETERMINACION DE LAS CUENTAS TEORICAS
cs_t=actividades.iloc[0,1]*0.851
eu_t=actividades.iloc[1,1]*0.2831
eu_t_1=actividades.iloc[1,1]*0.0749
eu_t_2=actividades.iloc[1,1]*0.266
co_t_1=actividades.iloc[2,1]*0.9988
co_t_2=actividades.iloc[2,1]*1
am_t=actividades.iloc[3,1]*0.357
cuentas_t=[]
cuentas_t.append(am_t)
cuentas_t.append(eu_t)
cuentas_t.append(eu_t_1)
cuentas_t.append(eu_t_2)
cuentas_t.append(cs_t)
cuentas_t.append(co_t_1)
cuentas_t.append(co_t_2)   
# DETERMINACION DE LA EFICIENCIA SIMULADA
eficiencia_d=np.zeros((3,7))
for j in range(areas.shape[1]):
    for i in range(areas.shape[0]):
        eficiencia_d[i,j]=areas[i,j]/cuentas_t[j]*100
efi_simulada=pd.DataFrame(data=eficiencia_d,columns=['Am-241 59 keV','Eu-152 121 keV','Eu-152 244 keV',
                                                     'Eu-152 344 keV', 'Cs-137 662 keV', 'Co-60 1.17 MeV',
                                                     'Co-60 1.33 MeV'])
efi_experimental=pd.read_csv('eficiencia_experimental.csv')
error=np.zeros((3,7))
for j in range(7):
    for i in range(3):
        error[i,j]=np.absolute((efi_simulada.iloc[i,j]-efi_experimental.iloc[i,j])/efi_experimental.iloc[i,j]*100)
error_comparacion=pd.DataFrame(data=error,columns=['Am-241 59 keV','Eu-152 121 keV','Eu-152 244 keV',
                                                     'Eu-152 344 keV', 'Cs-137 662 keV', 'Co-60 1.17 MeV',
                                                     'Co-60 1.33 MeV'],
                               index=['5cm', '10cm', '15cm'])       

fig5=plt.figure(figsize=(5,5))
plt.plot(Energy,eu_e_n['15cm'],'xr',label='Experimental')
plt.plot(eu_s_r['15cm'][0],eu_s_r1['15cm']/b_eu[0],'-g',label='Simulado')
plt.legend()
plt.grid()
plt.title('15cm')
# graficar las eficiencias experimentales vs simuladas
e_cab=[59.5, 121.78,244.69,344.28,662, 1173, 1332]
fig1=plt.figure(figsize=(5,5),dpi=2000)
# eficiencias experimentales
plt.plot(e_cab,efi_experimental.iloc[0,:],'o-g',label='Experimental 5cm')
plt.plot(e_cab,efi_experimental.iloc[1,:],'o-b',label='Experimental 10cm')
plt.plot(e_cab,efi_experimental.iloc[2,:],'o-r',label='Experimental 15cm')
# eficiencias simuladas
plt.plot(e_cab,efi_simulada.iloc[0,:],':g',marker=10,label='Simulated 5cm')
plt.plot(e_cab,efi_simulada.iloc[1,:],':b',marker=10,label='Simulated 10cm')
plt.plot(e_cab,efi_simulada.iloc[2,:],':r',marker=10,label='Simulated 15cm')
#plt.title("Curva de calibración de eficiencias")
plt.xlabel("Energy $(keV)$")
plt.ylabel("FEPE (%)")
plt.grid()
plt.legend()
plt.ticklabel_format(style='sci', axis='y',scilimits=(0,0))
plt.savefig('FEPE_G4andexp.jpg')
fig7=plt.figure(figsize=(5,5))
plt.plot(Energy,co_e_n['5cm'])
plt.plot(co_s_r['5cm'][0],co_s_r1['5cm']/302)
plt.xlim(1,1.5)
# recalculo factor de correccion
cs_n1=np.loadtxt('NaIDetector.dat')
cs_n1=np.histogram(cs_n1,bins=1500)
cs_n1r=spectrum(cs_n1[1], cs_n1[0], 0.026)
t_csp1=t_simulacion(1e7,actividades.iloc[0,1]*0.85)
cs_n1_=cs_n1r[1]
for i in range(len(cs_n1r[1])):
    cs_n1_[i]=cs_n1_[i]/(t_csp1) 
max(cs_n1_)/max(cs_e_n['5cm'])
for i in range(len(cs_n1r[1])):
    cs_n1_[i]=cs_n1_[i]/239 
plt.plot(cs_n1r[0],cs_n1_)
plt.plot(Energy,cs_e_n['5cm'])
#plots auxiliares
plt.plot(Energy, am_e_n['5cm'])
plt.xlim((0.01,0.08))