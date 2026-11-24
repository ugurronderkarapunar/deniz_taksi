import os
import warnings
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st

warnings.filterwarnings("ignore")

MAX_JOBLIB_MB = 25.0

RESMI_TATILLER_AY_GUN = {
    (1, 1),   # Yılbaşı
    (4, 23),  # Ulusal Egemenlik ve Çocuk Bayramı
    (5, 1),   # Emek ve Dayanışma Günü
    (5, 19),  # Atatürk'ü Anma, Gençlik ve Spor Bayramı
    (7, 15),  # Demokrasi ve Millî Birlik Günü
    (8, 30),  # Zafer Bayramı
    (10, 29), # Cumhuriyet Bayramı
}

ZAMAN_BLOKLARI = {
    "tek_saat": None,
    "sabah_pik": (7, 9),
    "aksam_donus": (17, 20),
    "gece_gec": (22, 23),
    "yaz_sezonu_aksam": (18, 21),
    "tum_gun": (0, 23),
}

SARIMAX_ADAYLARI = [
    ((0, 1, 1), (0, 1, 1, 7)),
    ((1, 1, 1), (0, 1, 1, 7)),
    ((0, 1, 2), (0, 1, 1, 7)),
    ((1, 0, 1), (1, 1, 1, 7)),
]

ISKELE_TABAN_TALEP = {
    "bostancı": 16.0,
    "büyükada": 22.0,
    "kadıköy": 25.0,
    "karaköy": 20.0,
    "beşiktaş": 21.0,
    "üsküdar": 19.0,
    "eminönü": 18.0,
    "kabataş": 17.0,
    "bebek": 9.0,
    "arnavutköy": 7.0,
}


class DosyasizFinalSARIMAX:
  def __init__(self, baslangic="2024-01-01", bitis="2026-10-25"):
    self.baslangic = pd.to_datetime(baslangic).floor("D")
    self.bitis = pd.to_datetime(bitis).floor("D")
    self.gunler = pd.date_range(self.baslangic, self.bitis, freq="D")

  @staticmethod
  def _saat_carpani(saat, mod="tek_saat"):
    saat = int(saat)
    if mod == "sabah_pik":
      return 1.35
    if mod == "aksam_donus":
      return 1.45
    if mod == "yaz_sezonu_aksam":
      return 1.40
    if mod == "gece_gec":
      return 0.35
    if mod == "tum_gun":
      return 1.15
    if saat in (8, 18, 19):
      return 1.35
    if saat in (7, 9, 17, 20):
      return 1.10
    if saat >= 22 or saat <= 5:
      return 0.40
    return 0.85

  def _seri_uret(self, iskele, saat=18, mod="tek_saat"):
    iskele_norm = str(iskele).strip().lower()
    baz = ISKELE_TABAN_TALEP.get(iskele_norm, 12.0)
    saat_katsayi = self._saat_carpani(saat=saat, mod=mod)

    tohum = abs(hash((iskele_norm, int(saat), mod))) % (2**32)
    rng = np.random.default_rng(tohum)

    n = int(self.gunler.shape[0])
    hafta_sonu_katsayi = np.where(self.gunler.dayofweek.isin([5, 6]), 1.35, 0.90)
    mevsim_katsayi = 1.0 + 0.30 * np.sin(
        (2.0 * np.pi / 365.25) * (self.gunler.dayofyear - 80)
    )
    gurultu = rng.normal(loc=0.0, scale=max(1.2, baz * 0.22), size=n)

    degerler = np.clip(
        baz * saat_katsayi * hafta_sonu_katsayi * mevsim_katsayi + gurultu,
        0.0,
        None,
    )
    return pd.Series(degerler, index=self.gunler, name="yolcu_sayisi").astype(float)

  @staticmethod
  def _exog_uret(indeks, ozel_gunler=None, anomali_gunler=None, gelecek_bilinmeyen_mod=False):
    idx = pd.DatetimeIndex(indeks)
    ozel_set = set()
    if ozel_gunler:
      for g in ozel_gunler:
        dt = pd.to_datetime(g, errors="coerce")
        if pd.notna(dt):
          ozel_set.add(dt.floor("D"))

    anomali_set = set()
    if anomali_gunler and not gelecek_bilinmeyen_mod:
      for g in anomali_gunler:
        dt = pd.to_datetime(g, errors="coerce")
        if pd.notna(dt):
          anomali_set.add(dt.floor("D"))

    exog = pd.DataFrame(index=idx)
    exog["hafta_sonu"] = idx.dayofweek.isin([5, 6]).astype(int)
    exog["resmi_tatil"] = np.array(
        [1 if ((d.month, d.day) in RESMI_TATILLER_AY_GUN) else 0 for d in idx],
        dtype=int,
    )
    exog["ozel_gun"] = np.array(
        [1 if d in ozel_set else 0 for d in idx], dtype=int
    )
    exog["iptal_veya_dusuk_gun"] = np.array(
        [1 if d in anomali_set else 0 for d in idx], dtype=int
    )

    herhangi_tatil = ((exog["resmi_tatil"] + exog["ozel_gun"]) > 0).astype(int)
    exog["tatil_oncesi"] = herhangi_tatil.shift(-1, fill_value=0).astype(int)
    exog["tatil_sonrasi"] = herhangi_tatil.shift(1, fill_value=0).astype(int)
    return exog.astype(int)

  @staticmethod
  def _metrikler(gercek, tahmin, tolerans_k=5):
    g = np.asarray(gercek, dtype=float).ravel()
    t = np.clip(np.asarray(tahmin, dtype=float).ravel(), 0.0, None)
    mutlak_hata = np.abs(g - t)
    return {
        "RMSE": round(float(np.sqrt(np.mean((g - t) ** 2))), 2),
        "MAE": round(float(np.mean(mutlak_hata)), 2),
        "Bias": round(float(np.mean(t - g)), 2),
        f"±{tolerans_k}_Yolcu_%": round(float(np.mean(mutlak_hata <= tolerans_k) * 100.0), 1),
    }

  @staticmethod
  def _hafif_kaydet(paket, kayit_yolu, max_mb=MAX_JOBLIB_MB):
    boyut_mb = 0.0
    for sıkıştırma in (3, 6, 9):
      joblib.dump(paket, kayit_yolu, compress=sıkıştırma)
      boyut_mb = os.path.getsize(kayit_yolu) / (1024 * 1024)
      if boyut_mb <= max_mb:
        return kayit_yolu, round(boyut_mb, 3)

    raise ValueError(
        f"Kaydedilen dosya ({boyut_mb:.2f} MB) belirlenen {max_mb:.0f} MB sınırını aştı."
    )

  def egit_ve_kaydet(
      self,
      iskele,
      saat=18,
      mod="tek_saat",
      test_baslangic="2026-01-01",
      test_gun=7,
      tolerans_k=5,
      ozel_gunler=None,
      kayit_yolu="final_sarimax_model.joblib",
  ):
    iskele_temiz = str(iskele).strip()
    if not iskele_temiz:
      raise ValueError("Kalkış iskelesi boş olamaz.")

    ts = self._seri_uret(iskele=iskele_temiz, saat=saat, mod=mod)
    test_start = pd.to_datetime(test_baslangic).floor("D")
    test_end = test_start + pd.Timedelta(days=int(test_gun) - 1)

    train_y = ts[ts.index < test_start]
    test_y = ts[(ts.index >= test_start) & (ts.index <= test_end)]

    if int(train_y.shape[0]) < 30 or int(test_y.shape[0]) == 0:
      raise ValueError("Eğitim veya test penceresi için yeterli gün bulunamadı.")

    pozitif_train = train_y[train_y > 0]
    esik = max(1.0, float(pozitif_train.quantile(0.25)) * 0.5) if int(pozitif_train.shape[0]) > 0 else 1.0
    train_anomali = set(train_y[train_y < esik].index)

    train_x = self._exog_uret(
        train_y.index,
        ozel_gunler=ozel_gunler,
        anomali_gunler=train_anomali,
        gelecek_bilinmeyen_mod=False,
    )
    test_x = self._exog_uret(
        test_y.index,
        ozel_gunler=ozel_gunler,
        anomali_gunler=None,
        gelecek_bilinmeyen_mod=True,
    )

    aday_kayitlari = []
    for order, s_order in SARIMAX_ADAYLARI:
      try:
        m = SARIMAX(
            train_y,
            exog=train_x,
            order=order,
            seasonal_order=s_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        p_test = np.clip(
            np.asarray(m.get_forecast(steps=int(test_y.shape[0]), exog=test_x).predicted_mean.values, dtype=float),
            0.0,
            None,
        )
        met = self._metrikler(test_y.to_numpy(dtype=float), p_test, tolerans_k=tolerans_k)
        aday_kayitlari.append({
            "SARIMAX_Order": f"{order}{s_order}",
            **met,
            "AIC": round(float(m.aic), 1),
            "_order": order,
            "_s_order": s_order,
        })
      except Exception:
        continue

    karsilastirma_df = (
        pd.DataFrame(aday_kayitlari)
        .sort_values(["RMSE", "AIC"], na_position="last")
        .reset_index(drop=True)
    )
    en_iyi_order = karsilastirma_df.loc[0, "_order"]
    en_iyi_s_order = karsilastirma_df.loc[0, "_s_order"]
    en_iyi_etiket = karsilastirma_df.loc[0, "SARIMAX_Order"]

    pozitif_full = ts[ts > 0]
    esik_full = max(1.0, float(pozitif_full.quantile(0.25)) * 0.5) if int(pozitif_full.shape[0]) > 0 else 1.0
    full_anomali = set(ts[ts < esik_full].index)
    full_x = self._exog_uret(
        ts.index,
        ozel_gunler=ozel_gunler,
        anomali_gunler=full_anomali,
        gelecek_bilinmeyen_mod=False,
    )

    final_fit = SARIMAX(
        ts,
        exog=full_x,
        order=en_iyi_order,
        seasonal_order=en_iyi_s_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    paket = {
        "params": np.asarray(final_fit.params, dtype=float),
        "order": en_iyi_order,
        "s_order": en_iyi_s_order,
        "iskele": iskele_temiz,
        "saat": int(saat),
        "mod": mod,
        "ozel_gunler": [str(pd.to_datetime(g).date()) for g in (ozel_gunler or [])],
        "final_model": f"SARIMAX{en_iyi_etiket}",
        "son_veri_tarihi": str(ts.index[-1].date()),
        "exog_kolonlari": list(full_x.columns),
        "karsilastirma_df": karsilastirma_df[[
            c for c in ["SARIMAX_Order", "RMSE", "MAE", "Bias", f"±{tolerans_k}_Yolcu_%", "AIC"]
            if c in karsilastirma_df.columns
        ]],
    }
    kayit_yolu, boyut_mb = self._hafif_kaydet(paket, kayit_yolu, max_mb=MAX_JOBLIB_MB)
    paket["dosya_boyutu_mb"] = boyut_mb
    return paket, kayit_yolu

  def tahmin_et(
      self,
      iskele,
      ay,
      gun,
      saat,
      mod="tek_saat",
      ozel_gun_mu=False,
      model_dosyasi="final_sarimax_model.joblib",
  ):
    ay_int = int(ay)
    gun_int = int(gun)
    saat_int = int(saat)

    try:
      hedef_tarih = pd.Timestamp(year=2026, month=ay_int, day=gun_int)
    except ValueError as exc:
      raise ValueError(f"Geçersiz ay/gün değeri: {ay_int}/{gun_int}") from exc

    ozel_tarihler = [hedef_tarih] if ozel_gun_mu else None
    self.egit_ve_kaydet(
        iskele=iskele,
        saat=saat_int,
        mod=mod,
        ozel_gunler=ozel_tarihler,
        kayit_yolu=model_dosyasi,
    )

    yuklenen = joblib.load(model_dosyasi)
    ts = self._seri_uret(iskele=yuklenen["iskele"], saat=yuklenen["saat"], mod=yuklenen["mod"])
    pozitif_full = ts[ts > 0]
    esik_full = max(1.0, float(pozitif_full.quantile(0.25)) * 0.5) if int(pozitif_full.shape[0]) > 0 else 1.0
    full_anomali = set(ts[ts < esik_full].index)
    full_x = self._exog_uret(
        ts.index,
        ozel_gunler=yuklenen.get("ozel_gunler"),
        anomali_gunler=full_anomali,
        gelecek_bilinmeyen_mod=False,
    )[yuklenen["exog_kolonlari"]]

    hafif_fit = SARIMAX(
        ts,
        exog=full_x,
        order=yuklenen["order"],
        seasonal_order=yuklenen["s_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).filter(yuklenen["params"])

    son_tarih = pd.to_datetime(yuklenen["son_veri_tarihi"])
    fark_gun = int((hedef_tarih - son_tarih).days)
    adim = max(7, fark_gun if fark_gun > 0 else 7)
    gelecek_idx = pd.date_range(son_tarih + pd.Timedelta(days=1), periods=adim, freq="D")

    fc_x = self._exog_uret(
        gelecek_idx,
        ozel_gunler=ozel_tarihler,
        anomali_gunler=None,
        gelecek_bilinmeyen_mod=True,
    )[yuklenen["exog_kolonlari"]]

    fc = hafif_fit.get_forecast(steps=adim, exog=fc_x)
    pred = np.round(np.clip(np.asarray(fc.predicted_mean.values, dtype=float), 0.0, None)).astype(int)
    ci = fc.conf_int(alpha=0.05)
    low = np.round(np.clip(np.asarray(ci.iloc[:, 0].values, dtype=float), 0.0, None)).astype(int)
    high = np.round(np.clip(np.asarray(ci.iloc[:, 1].values, dtype=float), 0.0, None)).astype(int)

    idx_pos = int(gelecek_idx.get_loc(hedef_tarih)) if hedef_tarih in gelecek_idx else 0
    saat_etiket = f"{saat_int:02d}:00" if mod == "tek_saat" else f"{mod} ({saat_int:02d}:00)"

    hedef_sonuc = {
        "kalkış_iskelesi": str(iskele).strip(),
        "tarih": hedef_tarih.strftime("%Y-%m-%d"),
        "ay": ay_int,
        "gün": gun_int,
        "saat": saat_etiket,
        "model": yuklenen["final_model"],
        "kayit_dosyasi": model_dosyasi,
        "dosya_boyutu_mb": yuklenen.get("dosya_boyutu_mb", round(os.path.getsize(model_dosyasi) / (1024 * 1024), 3)),
        "tahmin_yolcu": int(pred[idx_pos]),
        "alt_95": int(low[idx_pos]),
        "ust_95": int(high[idx_pos]),
    }

    pencere_df = pd.DataFrame({
        "tarih": gelecek_idx[:7].strftime("%Y-%m-%d"),
        "gun": gelecek_idx[:7].day_name(),
        "saat": saat_etiket,
        "tahmin_yolcu": pred[:7],
        "alt_95": low[:7],
        "ust_95": high[:7],
    })
    return hedef_sonuc, yuklenen["karsilastirma_df"], pencere_df


st.set_page_config(page_title="İskele Yolcu Tahmini (SARIMAX)", layout="centered")
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
    iskele = st.text_input("Kalkış İskelesi", value="bostancı")

  col1, col2, col3 = st.columns(3)
  with col1:
    ay = st.number_input("Ay (1-12)", min_value=1, max_value=12, value=12, step=1)
  with col2:
    gun = st.number_input("Gün (1-31)", min_value=1, max_value=31, value=5, step=1)
  with col3:
    saat = st.number_input("Saat (0-23)", min_value=0, max_value=23, value=18, step=1)

  mod = st.selectbox("Saat / Dilim Modu", list(ZAMAN_BLOKLARI.keys()), index=0)
  ozel_gun_mu = st.checkbox("Bu tarih özel gün / etkinlik / tatil etkisi taşısın mı?", value=False)
  model_dosyasi = st.text_input("Joblib Kayıt Dosyası", value="final_sarimax_model.joblib")
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

  st.subheader("SARIMAX Aday Karşılaştırması (Temel Metrik: RMSE)")
  st.dataframe(karsilastirma_df, use_container_width=True, hide_index=True)

  st.subheader("7 Günlük Tahmin Penceresi")
  st.dataframe(pencere_df, use_container_width=True, hide_index=True)
  st.line_chart(pencere_df.set_index("tarih")[["tahmin_yolcu", "ust_95"]], height=260)
