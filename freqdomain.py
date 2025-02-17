import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwpy.plot import Plot
from scipy import signal

from helper import makesine, make_audio_file, plot_signal

# -- Need to lock plots to be more thread-safe
from matplotlib.backends.backend_agg import RendererAgg
lock = RendererAgg.lock

cropstart = 1.0
cropend   = 1.05


def showfreqdomain():

    st.markdown("""
Se pensiamo a una rappresentazione di un segnale audio, ci verrà sicuramente in mente il 
grafico che mostra l'ampiezza del suono al variare del tempo (grafico nel **dominio del tempo**).
Un passo fondamentale di molti algoritmi di elaborazione dei segnali è quello di trasformare i 
dati in una diversa rappresentazione nel **dominio della frequenza**. 

ALLA RICERCA DI TRE NOTE

Il segnale qui sotto è la somma di 3 segnali di frequenze diverse. 
Riesci a ricostruire le loro proprietà (ampiezza e frequenza)?
""")

    st.markdown("#### Segnale target:")

    sig1 = makesine(150, 4, False)
    sig2 = makesine(200, 2, False)
    sig3 = makesine(350, 1, False)
    
    totalsignal = sig1+sig2+sig3
    plot_signal(totalsignal, color_num=1)

    st.audio(make_audio_file(totalsignal), format='audio/wav')

    st.markdown("""
    L'asse x rappresenta il tempo, l'asse y rappresenta la pressione dell'aria che 
    colpisce l'orecchio misurata in ogni intervallo temporale. In particolare, questo file
    audio è campionato a *32000 Hz* (32mila misurazioni al secondo, ossia una misurazione
    ogni circa 30 microsecondi). Questo tipo grafico può essere semplice da intuire, ma non 
    è ideale per visualizzare le proprietà del segnale. 
    
    Possiamo invece utilizzare un processo noto come trasformata di Fourier per convertire 
    il segnale dal *dominio del tempo* al *dominio della frequenza*. 
    
    :point_right: **Converti il segnale target al dominio della frequenza.**
    """)

    showfreq = st.checkbox('Applica la trasformata di Fourier al segnale target', value=False)

    if showfreq:
        freqdomain = totalsignal.fft()

        source = pd.DataFrame({
            'Frequency (Hz)': freqdomain.frequencies,
            'Amplitude': np.abs(freqdomain.value),
            'color':['#1f77b4', '#ff7f0e'][1]
        })

        chart = alt.Chart(source).mark_line().encode(
            alt.X('Frequency (Hz)',
                  scale=alt.Scale(
                      domain=(0, 400),
                      clamp=True)),
            alt.Y('Amplitude:Q',
                  scale=alt.Scale(
                      domain=(-0, 5),
                      clamp=True)),
            color=alt.Color('color', scale=None)
        ).properties(title='Segnale target nel dominio della frequenza')

        st.altair_chart(chart, use_container_width=True)
            
        st.markdown("""
        Il grafico nel **dominio della frequenza** mostra le sotto-componenti che contribuiscono al segnale target:

        * Quali sono le 3 frequenze?  
        * Qual è la loro ampiezza?
        """)


    st.markdown("""
    :point_right: **Prova a ricostruire il segnale sommando le tre componenti.**
    """)

    st.markdown("#### Componente 1")
    freq1 = st.slider("Frequenza (Hz)", 100, 400, 100, step=10)
    amp1 = st.number_input("Ampiezza", 0, 5, 0, key='amp1slider')

    with lock:
        guess1 = makesine(freq1, amp1)
    
    st.markdown("#### Componente 2")
    freq2 = st.slider("Frequenza (Hz)", 100, 400, 150, step=10)
    amp2 = st.number_input("Ampiezza", 0, 5, 0, key='amp2slider')

    with lock:
        guess2 = makesine(freq2, amp2)
    
    st.markdown("#### Componente 3")
    freq3 = st.slider("Frequenza (Hz)", 100, 400, 200, step=10)
    amp3 = st.number_input("Ampiezza", 0, 5, 0, key='amp3slider')

    with lock:
        guess3 = makesine(freq3, amp3)

    st.markdown("### Somma:")
    
    guess  = guess1 + guess2 + guess3

    chart1 = plot_signal(guess, color_num=0, display=False)
    chart2 = plot_signal(totalsignal, color_num=1, display=False)
    chart = (chart2 + chart1).properties(title='Segnale target (arancione) & Somma (blu)')
    st.altair_chart(chart, use_container_width=True)
        
    mismatch = (totalsignal.crop(cropstart, cropend) - guess.crop(cropstart, cropend)).value.max()
    # st.write(mismatch)

    if mismatch < 0.1:
        st.markdown("### Perfetto!!  :trophy:")
        st.balloons()
    elif mismatch < 3:
        st.markdown("### Quasi!")    
    
    st.markdown("#### Segnale target")
    st.audio(make_audio_file(totalsignal), format='audio/wav')

    st.markdown("#### Segnale ottenuto sommando i 3 contributi")
    st.audio(make_audio_file(guess), format='audio/wav')
    
    
    # -- Close all open figures
    plt.close('all')
