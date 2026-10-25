# Camera catalog research brief (VYVAR, CAMERA-PROFILE-2026-10)

VYVAR is a differential-photometry pipeline. It needs a catalog of **CMOS astronomy cameras** (cooled
and uncooled cameras sold for astronomy / scientific imaging, used for photometry). No CCD cameras, no
DSLR / mirrorless, no planetary-only webcams unless they are commonly used for photometry (e.g. ZWO
ASI174MM, ASI290MM, ASI432MM are used for occultation / fast photometry - include those).

Your job: produce the COMPLETE list of current and discontinued-but-common CMOS models of the vendors
assigned to you, with technical data, every number backed by a source URL. Use WebSearch / WebFetch
(manufacturer product pages, manuals, spec sheets PDFs, sensor-analysis pages from the manufacturer).
Do not invent numbers. If a value is not found, leave it out (null). Never guess the binning convention.

## Output

Write ONE JSON file to the path given in your task: `{"vendor_group": "...", "models": [ ... ]}`.
Each model:

```json
{
  "key": "zwo_asi2600mm_pro",            // lowercase, vendor_model, [a-z0-9_] only, unique
  "vendor": "ZWO",
  "model": "ASI2600MM Pro",
  "sensor": "Sony IMX571",
  "color": "mono",                       // "mono" or "osc"
  "bayer": null,                         // "RGGB"/"BGGR"/"GRBG"/"GBRG" for osc when documented, else null
  "aliases": ["ZWO ASI2600MM Pro", "ASI2600MM Pro", "ASI2600MM"],   // strings drivers write in INSTRUME / CAMERA
  "header_gain": "setting",              // "setting" if the driver writes the gain SETTING (ZWO, QHY, Player One, ToupTek...),
                                         // "e_per_adu" only if documented that the GAIN card is e-/ADU (e.g. Moravian SIPS)
  "cooled": true,
  "discontinued": false,
  "sources": ["https://..."],            // product page / manual used
  "modes": [
    {
      "key": "default",                  // readout mode key: "default", or e.g. "photographic", "high_gain", "extended_full_well", "11mp", "47mp"
      "name": "Default (16-bit)",
      "pixel_um": 3.76,
      "width": 6248, "height": 4176,     // effective / output resolution in px
      "adc_bits": 16,
      "binning": null,                   // ONLY if the manufacturer documents it: "sum" | "average" | "charge"; else null
      "binning_source": null,            // quote/URL when binning is set
      "curve": [                         // only documented numeric points, per native pixel (bin 1)
        {"setting": 100, "e_per_adu": 1.0, "source": "ZWO: unity gain at 100 (https://...)"},
        {"setting": 0, "e_per_adu": 0.78, "read_noise_e": 3.5, "source": "ZWO manual table ... (https://...)"}
      ],
      "full_well_e": 50000,              // optional, at lowest gain, if documented
      "read_noise_range_e": [1.0, 3.5]   // optional, as published
    }
  ]
}
```

Rules for the curve:
- A point needs a gain SETTING value (the number the driver uses) and at least one of e_per_adu /
  read_noise_e. "Unity gain at setting X" is a valid point (e_per_adu 1.0).
- Values read from a published chart only if the manufacturer prints the number (text or table);
  do not read values off a plot.
- Each point has its own short source text including the URL.

Include every model of your vendors (mono and color separately - they are different keys). Typical
count per big vendor: 30-80 models. Quality over speed, but be complete. ASCII only in the JSON.
At the end, report: number of models, how many have at least one curve point, and anything unsure.
