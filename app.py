# -- Use Agg backend to be thread safe
import matplotlib as mpl
mpl.use("agg")

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.api import fetch_event_json
from copy import deepcopy
import io
from scipy import signal
from scipy.io import wavfile
from freqdomain2 import showfreqdomain

# -- Need to lock plots to be more thread-safe
from matplotlib.backends.backend_agg import RendererAgg
lock = RendererAgg.lock

# -- Helper functions in this git repo
from helper import *

apptitle = 'Tutorial di Elaborazione dei Segnali'

st.set_page_config(page_title=apptitle, page_icon=":headphones:",
                               initial_sidebar_state='collapsed')

# Title the app
st.title(apptitle)

fs = 32000
noisedt = 8
noise = deepcopy(makewhitenoise(fs, noisedt))

#-- Try to color the noise
noisefreq = noise.fft()
color = 1.0 / (noisefreq.frequencies)**2
indx = np.where(noisefreq.frequencies.value < 30)
color[indx] = 0  #-- Apply low frequency cut-off at 30 Hz

#-- Red noise in frequency domain
weightedfreq = noisefreq * color.value

# -- Try returning to time domain
colorednoise = weightedfreq.ifft()

###
# -- Inject the signal
###
secret = TimeSeries.read('eff.wav')

# -- Normalize and convert to float
secret -= secret.value[0]  #-- Remove constant offset
secret = np.float64(secret)
secret = secret/np.max(np.abs(secret)) * 1*1e-7   #-- Set amplitude
secret.t0 = 2

volume = st.sidebar.radio("Volume", ["Default", "Louder"])

if volume == 'Louder':
    maze = colorednoise.inject(10*secret)
else:
# -- Might be useful to make easier to hear option
    maze = colorednoise.inject(secret)


# -------
# Begin Display Here
# -------
st.markdown("## Introduzione")

st.markdown("""
In questa laboratorio, cercheremo di trovare un suono nascosto nel rumore.
Per farlo, eserciteremo alcuni concetti di elaborazione del segnale (*signal processing*):

* Plot nel dominio del tempo e della frequenza
* Filtraggio ad alta frequenza (*highpass*) e passa-banda (*passband*)
* Whitening
""")

sectionnames = [
                'Introduzione dominio della frequenza (*frequency domain*)',
                'Rumore Bianco (*white noise*)',
                'Rumore Rosso (*red noise*)',
                'Suono Nascosto',
                'Whitening',
                'Dati di Onde Gravitazionali',
]

def headerlabel(number):
    return "{0}: {1}".format(number, sectionnames[number-1])
    
page = st.radio('Sezione:', [1,2,3,4,5,6], format_func=headerlabel)

st.markdown("## {}".format(headerlabel(page)))

if page==1:
    
    showfreqdomain()
    
if page==2:

    # White Noise
    
    st.markdown("""
    Il **rumore bianco** è un tipo di rumore statico che ha circa la
    stessa ampiezza a tutte le frequenze. In altre parole, ha una
    distribuzione di energia uniforme. Il rumore bianco viene utilizzato
    in diverse applicazioni, come il test delle prestazioni dei sistemi
    audio o la generazione di numeri casuali.
    
    Di seguito, viene rappresentato **lo stesso segnale in tre modi diversi**:
    
    * Nel dominio del tempo
    * Nel dominio delle frequenze
    * Un file audio
    """)

    st.markdown("### Dominio del tempo")

    st.markdown("""
    L'asse x rappresenta un valore di tempo, l'asse y l'ampiezza del segnale. 
    Per una registrazione audio, l'ampiezza del segnale corrisponde alla *quantità 
    di pressione* percepita sulla membrana del registratore (o sul timpano) in ogni intervallo temporale. 
    Per un'onda gravitazionale, l'ampiezza del segnale corrisponde alla *deformazione*
    (o variazione percentuale della lunghezza) delle braccia del rivelatore.
    """)

    with lock:
        tplot = noise.plot(ylabel='Pressure')
        st.pyplot(tplot)
    
    st.markdown("### Dominio della frequenza")

    st.markdown("""
    L'asse x rappresenta un valore di frequenza, l'asse y l'ampiezza (o similmente, la
    densità spettrale di ampiezza) del segnale per ogni frequenza.
    Poiché il rumore bianco ha circa la stessa ampiezza a ogni frequenza, questo 
    grafico è approssimativamente piatto.
    """)

    with lock:
        figwn = noise.asd(fftlength=1).plot(ylim=[1e-10, 1], ylabel='Amplitude Spectral Density')
        st.pyplot(figwn)

    st.markdown("### Player audio")
    st.markdown("""
    :point_right: **Usa il player audio per ascoltare un segnale di rumore bianco.**
    """)
    
    st.audio(make_audio_file(noise), format='audio/wav')

    st.markdown("")

    
if page == 3:

    # st.markdown("## 3: Red Noise")
    
    st.markdown("""
    Il **rumore rosso** ha più potenza alle basse frequenze.
    

    Immaginare il rumore random a diverse frequenze può essere difficile da comprendere. 
    
    Un esempio semplice può essere una foresta, in cui i versi di alcuni animali hanno 
    frequenze tipicamente alte (es. uccelli), mentre quelle di altri più basse (es. leoni).
    Se il loro contributo è simile si ha rumore bianco, se il contributo al rumore dei leoni
    è maggiore, si ha rumore rosso.

    Anche negli strumenti LIGO e Virgo ci sono diverse fonti di rumore. Tipicamente, gli 
    oggetti che vibrano alle basse frequenze contribuiscono al rumore alle basse frequenze,
    come i moti sismici. A frequenze più alte il rumore può essere generato dalle numerose
    parti strumentali dell'interferometro (come specchi e tavoli ottici).

    """)

    ###
    # -- Show red noise with signal
    ###

    st.markdown("Nel dominio del tempo il rumore rosso sembra completamente random...")

    with lock:
        figrnt = maze.plot(ylabel='Pressure')
        st.pyplot(figrnt)

    st.markdown("... ma nel dominio della frequenza si nota che c'è molta più potenza alle basse frequenze.")

    with lock:
        figrn = maze.asd(fftlength=1).plot(ylabel='Amplitude Spectral Density', ylim=[1e-11, 1e-4], xlim=[30, fs/2])
        st.pyplot(figrn)
        
    st.audio(make_audio_file(maze), format='audio/wav')
    st.markdown("""
    :point_right: **Come cambia rispetto al rumore bianco?**
    """)

if page == 4:

    # ----
    # Try to recover the signal
    # ----
    # st.markdown("## 4: Find the Secret Sound")
    
    st.markdown("""
    L'audio del rumore rosso ascoltato in precedenza non contiene solo rumore: c'è
    anche un segnale nascosto! Il rumore alle basse frequenze impedisce l'ascolto,
    ma ci sono molte tecniche per ripulire il segnale!
    
    All'ordine zero, ciò di cui abbiamo bisogno è un modo per eliminare parte del 
    rumore a basse frequenza, mantenendo la parte ad alta frequenza. 
    In elaborazione dei segnali, questo è noto come **filtro passa-alto** (highpass)
     - un filtro che rimuove i suoni a bassa frequenza e mantiene i suoni ad alta frequenza. 
    Il termine **frequenza di taglio** segna il confine al di sotto del quale le frequenze 
    vengono rimosse. 
    
    :point_right: **Regola la frequenza di taglio per trovare il suono nascosto.**

    """)

    lowfreq = st.slider("High pass filter cutoff frequency (Hz)", 0, 6000, 0, step=100)
    if lowfreq == 0: lowfreq=1

    highpass = maze.highpass(lowfreq)
    #st.pyplot(highpass.plot())

    with lock:
        fighp = highpass.asd(fftlength=1).plot(ylabel='Amplitude Spectral Density',
                                           ylim=[1e-12, 1e-5],
                                           xlim=[30, fs/2]
                                           )
        ax = fighp.gca()
        ax.axvspan(1, lowfreq, color='red', alpha=0.3, label='Removed by filter')
        st.pyplot(fighp)

    st.audio(make_audio_file(highpass), format='audio/wav')

    st.markdown("Can you hear the sound now?  What value of the cutoff frequency makes it easiest to hear?")

    st.markdown("")
    needhint = st.checkbox("Need a hint?", value=False)

    if needhint:

        st.markdown("""Here is the secret sound.  Can you find it hidden in the
        red noise above?
        """)

        st.audio(make_audio_file(secret), format='audio/wav')

        st.markdown("""You can also make the sound easier to hear by 
        clicking the 'Louder' option in the menu at left
        """)
        
if page == 5:
    # st.markdown("## 5: Whitening")

    st.markdown("""
    **Whitening** is a process that re-weights a signal, so that all
    frequency bins have a nearly equal amount of noise.  In our example,
    it is hard to hear the signal, because all of the low-frequency 
    noise covers it up.  By whitening the data,
    we can prevent the low-frequency noise from dominating what we hear. 
    
    :point_right: **Use the checkbox to whiten the data**
    """)

    
    whiten = st.checkbox("Whiten the data?", value=False)

    if whiten:
        whitemaze = maze.whiten()
    else:
        whitemaze = maze

    st.markdown("""
    After whitening, you can see the secret sound in the time domain.  You 
    may also notice that the whitened signal gently fades in at the beginning,
    and 
    out at the end - this gentle turn on / turn off is due to 
    **windowning**, and is important to apply in many signal processing
    applications.
    """)
    
    st.pyplot(whitemaze.plot())

    with lock:
        figwh = whitemaze.asd(fftlength=1).plot(ylim=[1e-12, 1], xlim=[30,fs/2], ylabel='Amplitude Spectral Density')
        st.pyplot(figwh)
    
    st.audio(make_audio_file(whitemaze), format='audio/wav')

    st.markdown("""Try using the checkbox to whiten the data.  Is it 
    easier to hear the secret sound with or without whitening?
    """)


if page == 6:

    # st.markdown("## 6: Gravitational Wave Data")

    st.markdown("""
    Finally, we'll try what we've learned on some real 
    gravitational-wave data from LIGO, around the binary black 
    hole signal GW150914.  We'll add one more element: 
    a **bandpass filter**.  A bandpass filter uses both a low frequency
    cutoff and a high frequency cutoff, and only passes signals in the 
    frequency band between these values. 

    :point_right: **Try using a whitening filter and a band-pass filter to reveal the
    gravitational wave signal in the data below.**  
    """)

    detector = 'H1'
    t0 = 1126259462.4   #-- GW150914

    st.text("Detector: {0}".format(detector))
    st.text("Time: {0} (GW150914)".format(t0))
    strain = load_gw(t0, detector)
    center = int(t0)
    strain = strain.crop(center-14, center+14)

    # -- Try whitened and band-passed plot
    # -- Whiten and bandpass data
    st.subheader('Whitened and Bandbassed Data')

    lowfreqreal, highfreqreal = st.slider("Band-pass filter cutoff (Hz)",
                                          1, 2000, value=(1,2000) )

    makewhite = st.checkbox("Apply whitening", value=False)

    if makewhite:
        white_data = strain.whiten()
    else:
        white_data = strain

    bp_data = white_data.bandpass(lowfreqreal, highfreqreal)

    st.markdown("""
    With the right filtering, you might be able to see the signal in the time domain plot.
    """)

    with lock:
        fig3 = bp_data.plot(xlim=[t0-0.1, t0+0.1])
        st.pyplot(fig3)

    # -- PSD of whitened data
    # -- Plot psd
    with lock:
        psdfig = bp_data.asd(fftlength=4).plot(xlim=[10, 1800], ylabel='Amplitude Spectral Density')    
        ax = psdfig.gca()
        ax.axvspan(1, lowfreqreal, color='red', alpha=0.3, label='Removed by filter')
        ax.axvspan(highfreqreal, 1800, color='red', alpha=0.3, label='Removed by filter')
        st.pyplot(psdfig)

    # -- Audio
    st.audio(make_audio_file(bp_data.crop(t0-1, t0+1)), format='audio/wav')

    # -- Close all open figures
    plt.close('all')

    st.markdown("""With the right filtering, you might be able to hear
    the black hole signal.  It doesn't sound like much - just a quick thump.  
 """)

    st.markdown("")
    hint = st.checkbox('Need a hint?')

    if hint:

        st.markdown("""
        Hint: Try using a band pass from 30 to 400 Hz, with whitening on.
        This is similar to what was used for Figure 1 of the 
        [GW150914 discovery paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.061102), also shown below:
        """)
        
        st.image('https://journals.aps.org/prl/article/10.1103/PhysRevLett.116.061102/figures/1/large')



st.markdown("""## Credits
Questa app è riadattata da 
[streamlit-audio di Jonah Kanner](https://github.com/jkanner/streamlit-audio) e contiene dati di LIGO, Virgo, e GEO [https://gw-openscience.org](https://gw-osc.org).
""")
