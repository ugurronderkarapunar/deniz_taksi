import streamlit as st
from sarimax_model import (
    DosyasizFinalSARIMAX,
    ISKELE_TABAN_TALEP,
    MAX_JOBLIB_MB,
    ZAMAN_BLOKLARI,
)

st.set_page_config(
    page_title="İskele Yolcu Tahmini (SARIMAX)",
    layout="centered",
)

st.title("İskele Yolcu Tahmini (SARIMAX)")
st.caption(
    "Dosya yüklemeden kalkış iskelesi, ay, gün ve saat girerek tahmin alın. "
    f"Model `joblib` ile en fazla {MAX_JOBLIB_MB:.0f} MB sınırında kaydedilir ve tahminler 0 tabanında tutulur."
)

hazir_iskeleler = sorted(ISKELE_TABAN_TALEP.keys())

with st.form("tahmin_formu"):
  iskele_giris_modu = st.radio(
      "Kalkış İskelesi Girişi",
      ["Listeden Seç", "Manuel Yaz"],
      horizontal=True,
  )
  if iskele_giris_modu == "Listeden Seç":
    varsayilan_idx = hazir_iskeleler.index("bostancı") if "bostancı" in hazir_iskeleler else 0
    iskele = st.selectbox("Kalkış İskelesi", hazir_iskeleler, index=varsayilan_idx)
  else:
    iskele = st.text_input(
        "Kalkış İskelesi",
        value="bostancı",
        placeholder="Örn: bostancı, büyükada, kadıköy",
    )

  col1, col2, col3 = st.columns(3)
  with col1:
    ay = st.number_input("Ay (1-12)", min_value=1, max_value=12, value=12, step=1)
  with col2:
    gun = st.number_input("Gün (1-31)", min_value=1, max_value=31, value=5, step=1)
  with col3:
    saat = st.number_input("Saat (0-23)", min_value=0, max_value=23, value=18, step=1)

  mod = st.selectbox("Saat / Dilim Modu", list(ZAMAN_BLOKLARI.keys()), index=0)
  ozel_gun_mu = st.checkbox(
      "Bu tarih özel gün / etkinlik / tatil etkisi taşısın mı?",
      value=False,
  )
  model_dosyasi = st.text_input(
      "Joblib Kayıt Dosyası",
      value="final_sarimax_model.joblib",
  )

  gonder = st.form_submit_button("Modeli Eğit, Kaydet ve Tahmin Yap", type="primary")

if gonder:
  modelci = DosyasizFinalSARIMAX()
  with st.spinner("SARIMAX adayları sınanıyor, model joblib'e kaydediliyor ve tahmin üretiliyor..."):
    try:
      hedef_sonuc, karsilastirma_df, pencere_df = modelci.tahmin_et(
          iskele=iskele,
          ay=int(ay),
          gun=int(gun),
          saat=int(saat),
          mod=mod,
          ozel_gun_mu=ozel_gun_mu,
          model_dosyasi=model_dosyasi.strip() or "final_sarimax_model.joblib",
      )
    except Exception as exc:
      st.error(f"Tahmin sırasında hata oluştu: {exc}")
      st.stop()

  boyut_mb = float(hedef_sonuc.get("dosya_boyutu_mb", 0.0))
  boyut_kb = round(boyut_mb * 1024.0, 1)

  st.success(
      f"**{hedef_sonuc['kalkış_iskelesi']}** | **{hedef_sonuc['tarih']}** | "
      f"**{hedef_sonuc['saat']}** → Tahmini Yolcu: **{hedef_sonuc['tahmin_yolcu']}** "
      f"(`{hedef_sonuc['kayit_dosyasi']}`, {boyut_mb} MB / {boyut_kb} KB)"
  )

  m1, m2, m3 = st.columns(3)
  m1.metric("Tahmin Yolcu", hedef_sonuc["tahmin_yolcu"])
  m2.metric("Alt Sınır (%95)", hedef_sonuc["alt_95"])
  m3.metric("Üst Sınır (%95)", hedef_sonuc["ust_95"])

  st.caption(f"Seçilen Model: {hedef_sonuc['model']}")

  st.subheader("SARIMAX Aday Karşılaştırması (Temel Metrik: RMSE)")
  st.dataframe(karsilastirma_df, use_container_width=True, hide_index=True)

  st.subheader("7 Günlük Tahmin Penceresi")
  st.dataframe(pencere_df, use_container_width=True, hide_index=True)
  st.line_chart(pencere_df.set_index("tarih")[["tahmin_yolcu", "ust_95"]], height=260)
