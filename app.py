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
from freqdomain import showfreqdomain

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
secret = TimeSeries.read('sound.wav')

# -- Normalize and convert to float
secret -= secret.value[0]  #-- Remove constant offset
secret = np.float64(secret)
secret = secret/np.max(np.abs(secret)) * 1.*1e-8   #-- Set amplitude
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
In questo laboratorio esploreremo alcuni concetti di elaborazione del segnale 
(*signal processing*) e cercheremo di rivelare un suono nascosto nel rumore.

Concetti chiave:
* Grafici nel dominio del tempo e della frequenza
* Filtraggio ad alta frequenza (*highpass*) e passa-banda (*passband*)
* Whitening
""")

sectionnames = [
                'Introduzione al dominio della frequenza (*frequency domain*)',
                'Rumore Bianco (*white noise*)',
                'Rumore Rosso (*red noise*)',
                'Suono Nascosto',
                'Whitening',
                'Onde Gravitazionali',
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
    Il **rumore bianco** è un tipo di rumore statico che ha circa la stessa ampiezza a tutte 
    le frequenze (in altre parole, ha una distribuzione di energia uniforme). Viene utilizzato
    in diverse applicazioni, come il test delle prestazioni dei sistemi audio o la generazione 
    di numeri casuali.
    
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
    (o variazione percentuale della lunghezza) dei bracci del rivelatore.
    """)

    with lock:
        tplot = noise.plot(ylabel='Pressure')
        st.pyplot(tplot)
    
    st.markdown("### Dominio della frequenza")

    st.markdown("""
    L'asse x rappresenta un valore di frequenza, l'asse y l'ampiezza (o similmente, la
    densità spettrale di ampiezza) del segnale per ogni frequenza.
    Poiché il rumore bianco ha circa la stessa ampiezza a ogni frequenza, la linea è 
    approssimativamente piatta.
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
    
    Com'è fatto un rumore random, ma diverso a frequenze diverse? Immaginiamo una foresta 
    popolata da tantissimi animali che emettono versi a frequenze diverse (per esempio, animali 
    piccoli come gli uccelli alle alte frequenze e animalii grandi come i leoni alle basse 
    frequenze). Se il loro contributo è uniforme si ha rumore bianco, se il contributo dei 
    leoni è molto maggiore, si ha rumore rosso.

    Negli strumenti LIGO e Virgo ci sono diverse fonti di rumore. Tipicamente, gli 
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
    anche un segnale nascosto! Il rumore alle basse frequenze ci impedisce di rivelarlo,
    ma ci sono molte tecniche per ripulire il segnale!
    
    Come primo passo, ciò di cui abbiamo bisogno è un modo per eliminare parte del suono 
    a basse frequenza, mantenendo la parte ad alta frequenza. Questa procedura è nota 
    come **filtro passa-alto** (highpass). 
    Il termine **frequenza di taglio** segna il confine al di sotto del quale le frequenze 
    vengono rimosse. 
    
    :point_right: **Regola la frequenza di taglio per trovare il suono nascosto.**

    """)

    lowfreq = st.slider("High pass filter cutoff frequency (Hz)", 0, 3000, 0, step=100)
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

    st.markdown("")
    needhint = st.checkbox("Suggerimento?", value=False)

    if needhint:

        st.markdown("""Questo è il suono nascosto nel rumore!
        """)

        st.audio(make_audio_file(secret), format='audio/wav')

        st.markdown("""Si può anche rendere più semplice da trovare cliccando l'opzione 'Louder'
        nel menù a sinistra.
        """)
        
if page == 5:
    # st.markdown("## 5: Whitening")

    st.markdown("""
    Il **whitening** è un processo che ripesa un segnale in modo tale che tutti gli intervalli
    di frequenza abbiano circa lo stesso livello di rumore. Nell'esempio precedente, la rivelazione
    del segnale è compromessa dall'alto rumore alle basse frequenze. Tuttavia, "sbiancando" il segnale 
    riusciamo a ridurre significativamente questo contributo.

    :point_right: **Ripulisci il suono con un metodo alternativo**
    """)

    
    whiten = st.checkbox("Applica whitening", value=False)

    if whiten:
        whitemaze = maze.whiten()
    else:
        whitemaze = maze

    st.markdown("""
    In seguito al whitening, emerge chiaramente il segnale del suono nascosto. Si può anche
    notare che il segnale processato sfuma gradualmente all'inizio e alla fine: questo è 
    dovuto all'utilizzo di una funzione finestra (*window/tapering function*) che è fondamentale
    per definire l'intervallo entro cui, via via, rinormalizzare il rumore.
    """)
    
    st.pyplot(whitemaze.plot())

    with lock:
        figwh = whitemaze.asd(fftlength=1).plot(ylim=[1e-12, 1], xlim=[30,fs/2], ylabel='Amplitude Spectral Density')
        st.pyplot(figwh)
    
    st.audio(make_audio_file(whitemaze), format='audio/wav')


if page == 6:

    # st.markdown("## 6: Gravitational Wave Data")

    st.markdown("""
    Infine, proviamo ad applicare i concetti di questo breve tutorial ad alcuni dari reali
    di onde gravitazionali. In particolare utilizzaremo il primissimo evento rivelato il 
    14 settembre 2015 (GW150914), generato dalla fusione di due buchi neri distanti più di 
    un miliardo di anni luce dalla Terra. 

    
    Aggiungeremo un ultimo concetto di eleborazione dei segnali: il *filtro passa-banda* 
    (**bandpass**), ossia un filtro che taglia le frequenze sotto una data soglia e quelle
    sopra a una data soglia (passa soltanto quelle intermedie). 

    :point_right: **Utilizza un filtro di whitening e un filtro passa-banda per rivelare
    il segnale di onde gravitazionali.**  
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
    st.subheader('Segnale processato')

    lowfreqreal, highfreqreal = st.slider("Band-pass filter cutoff (Hz)",
                                          1, 1300, value=(1,1300) )

    makewhite = st.checkbox("Applica whitening", value=False)

    if makewhite:
        white_data = strain.whiten()
    else:
        white_data = strain

    bp_data = white_data.bandpass(lowfreqreal, highfreqreal)

    st.markdown("""
    Con il filtraggio corretto si può far emergere il segnale nel grafico.
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
    hint = st.checkbox('Suggerimento?')

    if hint:

        st.markdown("""Prova ad applicare il whitening e utilizza un filtro passa-banda 
        da 30 a 400 Hz. Questo, in prima approssimazione, è lo stesso processo utilizzato per la
        Figura 1 dell'articolo scientifico sulla 
        [scoperta di GW150914](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.061102):
        """)
        
        st.image('https://journals.aps.org/prl/article/10.1103/PhysRevLett.116.061102/figures/1/large')



st.markdown("""## Credits
Questa app è riadattata da 
[streamlit-audio di Jonah Kanner](https://github.com/jkanner/streamlit-audio) e contiene dati di LIGO, Virgo, e GEO [https://gw-openscience.org](https://gw-osc.org).
""")
