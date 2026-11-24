import calendar
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st

warnings.filterwarnings("ignore")

MAX_JOBLIB_MB = 25.0

AY_ADLARI = {
    1: "Ocak",
    2: "Şubat",
    3: "Mart",
    4: "Nisan",
    5: "Mayıs",
    6: "Haziran",
    7: "Temmuz",
    8: "Ağustos",
    9: "Eylül",
    10: "Ekim",
    11: "Kasım",
    12: "Aralık",
}

AY_GUN_SAYILARI = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

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
    # Ana ve Adalar / Sahil iskeleleri
    "kadıköy": 25.0,
    "büyükada": 22.0,
    "beşiktaş": 21.0,
    "karaköy": 20.0,
    "üsküdar": 19.0,
    "eminönü": 18.0,
    "kabataş": 17.0,
    "bostancı": 16.0,
    "heybeliada": 15.0,
    "sirkeci": 14.5,
    "burgazada": 13.0,
    "moda": 12.5,
    "kınalıada": 11.0,
    "çengelköy": 10.5,
    "kuzguncuk": 9.5,
    "bebek": 9.0,
    "sarıyer": 8.5,
    "arnavutköy": 7.0,
    "kanlıca": 6.5,
    "emirgan": 6.0,
    "beykoz": 5.5,
    # Boğaz hattı diğer iskeleler
    "ortaköy": 12.0,
    "anadolu hisarı": 7.5,
    "küçüksu": 7.0,
    "beylerbeyi": 8.0,
    "aşiyan": 6.5,
    "istinye": 7.5,
    "çubuklu": 6.0,
    "paşabahçe": 6.0,
    "rumeli kavağı": 5.0,
    "anadolu kavağı": 5.5,
    # Haliç hattı iskeleleri
    "kasımpaşa": 8.5,
    "hasköy": 6.5,
    "sütlüce": 6.0,
    "eyüpsultan": 9.5,
    "balat": 7.5,
    "fener": 7.5,
    "ayvansaray": 6.0,
    # Anadolu yakası diğer iskeleler
    "maltepe": 10.0,
    "pendik": 7.5,
    "tuzla": 6.5,
}


def ay_icin_gun_sayisi(yil: int, ay: int) -> int:
  yil_int = int(yil)
  ay_int = int(ay)
  if ay_int == 2:
    return 29 if calendar.isleap(yil_int) else 28
  return AY_GUN_SAYILARI.get(ay_int, 31)


def saat_araligi_etiketi(saat: int, mod: str = "tek_saat") -> str:
  saat = int(saat)
  if mod in ZAMAN_BLOKLARI and ZAMAN_BLOKLARI[mod] is not None:
    bas, bit = ZAMAN_BLOKLARI[mod]
    return f"{bas:02d}:00 - {bit:02d}:59 ({mod})"
  return f"{saat:02d}:00 - {saat:02d}:59 (tek_saat)"


class DosyasizFinalSARIMAX:
  def __init__(self, baslangic="2024-01-01", bitis="2026-12-31"):
    self.baslangic = pd.to_datetime(baslangic).floor("D")
    self.bitis = pd.to_datetime(bitis).floor("D")
    self.gunler = pd.date_range(self.baslangic, self.bitis, freq="D")

  def _hedef_ufuk_penceresi(self, hedef_yil: int):
    yil_int = int(hedef_yil)
    egitim_sonu = pd.Timestamp(year=max(2025, yil_int - 1), month=12, day=31)
    test_baslangic = pd.Timestamp(year=yil_int, month=1, day=1)
    ufuk_bitis = pd.Timestamp(year=yil_int, month=12, day=31)
    gunler = pd.date_range(self.baslangic, egitim_sonu, freq="D")
    return gunler, test_baslangic, ufuk_bitis

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

  def _seri_uret(self, iskele, saat=18, mod="tek_saat", gunler=None):
    iskele_norm = str(iskele).strip().lower()
    if iskele_norm not in ISKELE_TABAN_TALEP:
      raise ValueError("Geçersiz kalkış iskelesi seçimi.")
    baz = ISKELE_TABAN_TALEP[iskele_norm]
    saat_katsayi = self._saat_carpani(saat=saat, mod=mod)

    aktif_gunler = pd.DatetimeIndex(gunler) if gunler is not None else self.gunler
    tohum = abs(hash((iskele_norm, int(saat), mod))) % (2**32)
    rng = np.random.default_rng(tohum)

    n = int(aktif_gunler.shape[0])
    hafta_sonu_katsayi = np.where(aktif_gunler.dayofweek.isin([5, 6]), 1.35, 0.90)
    mevsim_katsayi = 1.0 + 0.30 * np.sin(
        (2.0 * np.pi / 365.25) * (aktif_gunler.dayofyear - 80)
    )
    gurultu = rng.normal(loc=0.0, scale=max(1.2, baz * 0.22), size=n)

    degerler = np.clip(
        baz * saat_katsayi * hafta_sonu_katsayi * mevsim_katsayi + gurultu,
        0.0,
        None,
    )
    return pd.Series(degerler, index=aktif_gunler, name="yolcu_sayisi").astype(float)

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
    for sikistirma in (3, 6, 9):
      joblib.dump(paket, kayit_yolu, compress=sikistirma)
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
      yil=2027,
      mod="tek_saat",
      test_gun=7,
      tolerans_k=5,
      ozel_gunler=None,
      kayit_yolu="final_sarimax_model.joblib",
  ):
    iskele_temiz = str(iskele).strip().lower()
    if iskele_temiz not in ISKELE_TABAN_TALEP:
      raise ValueError("Geçersiz kalkış iskelesi seçimi.")

    yil_int = int(yil)
    gunler, test_start, ufuk_bitis = self._hedef_ufuk_penceresi(yil_int)
    ts = self._seri_uret(iskele=iskele_temiz, saat=saat, mod=mod, gunler=gunler)

    hedef_yil_gunler = pd.date_range(test_start, ufuk_bitis, freq="D")
    hedef_yil_seri = self._seri_uret(iskele=iskele_temiz, saat=saat, mod=mod, gunler=hedef_yil_gunler)

    train_y = ts
    test_y = hedef_yil_seri.iloc[: max(1, int(test_gun))]

    if int(train_y.shape[0]) < 30 or int(test_y.shape[0]) == 0:
      raise ValueError("Eğitim veya hedef yıl penceresi için yeterli gün bulunamadı.")

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
            "_abs_bias": abs(float(met["Bias"])),
            "AIC": round(float(m.aic), 1),
            "_order": order,
            "_s_order": s_order,
        })
      except Exception:
        continue

    # Seçim kriteri: en düşük mutlak Bias (|Bias|); eşitlikte RMSE ve AIC
    karsilastirma_df = (
        pd.DataFrame(aday_kayitlari)
        .sort_values(["_abs_bias", "RMSE", "AIC"], na_position="last")
        .reset_index(drop=True)
    )
    en_iyi_order = karsilastirma_df.loc[0, "_order"]
    en_iyi_s_order = karsilastirma_df.loc[0, "_s_order"]
    en_iyi_etiket = karsilastirma_df.loc[0, "SARIMAX_Order"]

    final_fit = SARIMAX(
        train_y,
        exog=train_x,
        order=en_iyi_order,
        seasonal_order=en_iyi_s_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    paket = {
        "params": np.asarray(final_fit.params, dtype=float),
        "order": en_iyi_order,
        "s_order": en_iyi_s_order,
        "secim_kriteri": "en_dusuk_bias",
        "iskele": iskele_temiz,
        "saat": int(saat),
        "hedef_yil": yil_int,
        "mod": mod,
        "ozel_gunler": [str(pd.to_datetime(g).date()) for g in (ozel_gunler or [])],
        "final_model": f"SARIMAX{en_iyi_etiket}",
        "egitim_baslangic": str(train_y.index[0].date()),
        "egitim_bitis": str(train_y.index[-1].date()),
        "ufuk_bitis": str(ufuk_bitis.date()),
        "exog_kolonlari": list(train_x.columns),
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
      yil=2027,
      mod="tek_saat",
      ozel_gun_mu=False,
      model_dosyasi="final_sarimax_model.joblib",
  ):
    yil_int = int(yil)
    ay_int = int(ay)
    gun_int = int(gun)
    saat_int = int(saat)

    try:
      hedef_tarih = pd.Timestamp(year=yil_int, month=ay_int, day=gun_int)
    except ValueError as exc:
      raise ValueError(
          f"Geçersiz tarih değeri: {gun_int} {AY_ADLARI.get(ay_int, ay_int)} {yil_int}"
      ) from exc

    ozel_tarihler = [hedef_tarih] if ozel_gun_mu else None
    paket, _ = self.egit_ve_kaydet(
        iskele=iskele,
        saat=saat_int,
        yil=yil_int,
        mod=mod,
        ozel_gunler=ozel_tarihler,
        kayit_yolu=model_dosyasi,
    )

    yuklenen = joblib.load(model_dosyasi)
    gunler, _, _ = self._hedef_ufuk_penceresi(yuklenen.get("hedef_yil", yil_int))
    ts = self._seri_uret(
        iskele=yuklenen["iskele"],
        saat=yuklenen["saat"],
        mod=yuklenen["mod"],
        gunler=gunler,
    )

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

    son_tarih = pd.to_datetime(yuklenen["egitim_bitis"])
    pencere_baslangic = hedef_tarih if hedef_tarih > son_tarih else (son_tarih + pd.Timedelta(days=1))
    pencere_son = pencere_baslangic + pd.Timedelta(days=6)
    fark_gun = int((pencere_son - son_tarih).days)
    adim = max(7, fark_gun)
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

    pred_series = pd.Series(pred, index=gelecek_idx)
    low_series = pd.Series(low, index=gelecek_idx)
    high_series = pd.Series(high, index=gelecek_idx)

    if hedef_tarih in pred_series.index:
      hedef_tahmin = int(pred_series.loc[hedef_tarih])
      hedef_low = int(low_series.loc[hedef_tarih])
      hedef_high = int(high_series.loc[hedef_tarih])
    else:
      hedef_tahmin = int(pred_series.iloc[0])
      hedef_low = int(low_series.iloc[0])
      hedef_high = int(high_series.iloc[0])

    pencere_idx = pd.date_range(pencere_baslangic, periods=7, freq="D")
    saat_etiket = saat_araligi_etiketi(saat_int, mod)

    hedef_sonuc = {
        "kalkış_iskelesi": str(iskele).strip().lower(),
        "tarih": hedef_tarih.strftime("%Y-%m-%d"),
        "yıl": yil_int,
        "ay": AY_ADLARI.get(ay_int, str(ay_int)),
        "gün": gun_int,
        "saat_araligi": saat_etiket,
        "eğitim_penceresi": f"{yuklenen['egitim_baslangic']} .. {yuklenen['egitim_bitis']}",
        "model": yuklenen["final_model"],
        "secim_kriteri": yuklenen.get("secim_kriteri", "en_dusuk_bias"),
        "kayit_dosyasi": model_dosyasi,
        "dosya_boyutu_mb": paket["dosya_boyutu_mb"],
        "tahmin_yolcu": hedef_tahmin,
        "alt_95": hedef_low,
        "ust_95": hedef_high,
    }

    pencere_df = pd.DataFrame({
        "tarih": pencere_idx.strftime("%Y-%m-%d"),
        "gun": pencere_idx.day_name(),
        "saat_araligi": saat_etiket,
        "tahmin_yolcu": [int(pred_series.get(d, pred_series.iloc[-1])) for d in pencere_idx],
        "alt_95": [int(low_series.get(d, low_series.iloc[-1])) for d in pencere_idx],
        "ust_95": [int(high_series.get(d, high_series.iloc[-1])) for d in pencere_idx],
    })
    return hedef_sonuc, yuklenen["karsilastirma_df"], pencere_df


@st.cache_data(show_spinner=False)
def onbellekli_tahmin_et(
    iskele: str,
    yil: int,
    ay: int,
    gun: int,
    saat: int,
    mod: str,
    ozel_gun_mu: bool,
    model_dosyasi: str,
):
  modelci = DosyasizFinalSARIMAX()
  return modelci.tahmin_et(
      iskele=iskele,
      yil=yil,
      ay=ay,
      gun=gun,
      saat=saat,
      mod=mod,
      ozel_gun_mu=ozel_gun_mu,
      model_dosyasi=model_dosyasi,
  )


st.set_page_config(page_title="İskele Yolcu Tahmini (SARIMAX)", layout="centered")
st.title("İskele Yolcu Tahmini (SARIMAX)")
st.caption(
    "Kalkış iskelesi, hedef yıl, ay, gün ve saat aralığı seçerek tahmin alın. "
    f"Aday modeller en düşük mutlak Bias kriteriyle seçilir, `joblib` ile en fazla {MAX_JOBLIB_MB:.0f} MB sınırında kaydedilir."
)

iskele_secenekleri = list(ISKELE_TABAN_TALEP.keys())
yil_secenekleri = [2026, 2027, 2028, 2029]
ay_secenekleri = list(AY_ADLARI.keys())

col_y, col_a, col_g = st.columns(3)
with col_y:
  yil = st.selectbox(
      "Hedef Yıl",
      options=yil_secenekleri,
      index=1,
  )
with col_a:
  ay = st.selectbox(
      "Ay",
      options=ay_secenekleri,
      index=11,
      format_func=lambda m: AY_ADLARI[m],
  )
with col_g:
  max_gun = ay_icin_gun_sayisi(int(yil), int(ay))
  gun = st.selectbox(
      "Gün",
      options=list(range(1, max_gun + 1)),
      index=min(4, max_gun - 1),
  )

saat_secenekleri = []
for blok_adi, aralik in ZAMAN_BLOKLARI.items():
  if aralik is not None:
    bas, bit = aralik
    saat_secenekleri.append((f"{bas:02d}:00 - {bit:02d}:59 ({blok_adi})", blok_adi, bas))
for h in range(24):
  saat_secenekleri.append((f"{h:02d}:00 - {h:02d}:59 (tek_saat)", "tek_saat", h))

with st.form("tahmin_formu"):
  iskele = st.selectbox(
      "Kalkış İskelesi",
      options=iskele_secenekleri,
      index=iskele_secenekleri.index("bostancı") if "bostancı" in iskele_secenekleri else 0,
  )
  secilen_saat_araligi = st.selectbox(
      "Saat Aralığı",
      options=saat_secenekleri,
      index=1,
      format_func=lambda item: item[0],
  )
  ozel_gun_mu = st.checkbox(
      "Bu tarih özel gün / etkinlik / tatil etkisi taşısın mı?",
      value=False,
  )
  model_dosyasi = st.text_input(
      "Joblib Kayıt Dosyası",
      value="final_sarimax_model.joblib",
  )
  gonder = st.form_submit_button("En Düşük Bias ile Eğit ve Tahmin Yap", type="primary")

if gonder:
  _, mod_secimi, saat_degeri = secilen_saat_araligi
  with st.spinner("En düşük mutlak Bias kriteriyle SARIMAX adayları sınanıyor ve model kaydediliyor..."):
    try:
      hedef_sonuc, karsilastirma_df, pencere_df = onbellekli_tahmin_et(
          iskele=iskele,
          yil=int(yil),
          ay=int(ay),
          gun=int(gun),
          saat=int(saat_degeri),
          mod=mod_secimi,
          ozel_gun_mu=ozel_gun_mu,
          model_dosyasi=model_dosyasi.strip() or "final_sarimax_model.joblib",
      )
    except Exception as exc:
      st.error(f"Tahmin sırasında hata oluştu: {exc}")
      st.stop()

  boyut_mb = float(hedef_sonuc.get("dosya_boyutu_mb", 0.0))
  boyut_kb = round(boyut_mb * 1024.0, 1)

  st.success(
      f"**{hedef_sonuc['kalkış_iskelesi']}** | **{hedef_sonuc['gün']} {hedef_sonuc['ay']} {hedef_sonuc['yıl']}** | "
      f"**{hedef_sonuc['saat_araligi']}** → Tahmini Yolcu: **{hedef_sonuc['tahmin_yolcu']}** "
      f"(`{hedef_sonuc['kayit_dosyasi']}`, {boyut_mb} MB / {boyut_kb} KB)"
  )

  m1, m2, m3 = st.columns(3)
  m1.metric("Tahmin Yolcu", hedef_sonuc["tahmin_yolcu"])
  m2.metric("Alt Sınır (%95)", hedef_sonuc["alt_95"])
  m3.metric("Üst Sınır (%95)", hedef_sonuc["ust_95"])

  st.caption(
      f"Seçilen Model: {hedef_sonuc['model']} | Kriter: En Düşük |Bias| | "
      f"Eğitim Penceresi: {hedef_sonuc['eğitim_penceresi']}"
  )

  st.subheader("SARIMAX Aday Karşılaştırması (Temel Metrik: En Düşük |Bias|)")
  st.dataframe(karsilastirma_df, use_container_width=True, hide_index=True)

  st.subheader("7 Günlük Tahmin Penceresi")
  st.dataframe(pencere_df, use_container_width=True, hide_index=True)
  st.line_chart(pencere_df.set_index("tarih")[["tahmin_yolcu", "ust_95"]], height=260)
