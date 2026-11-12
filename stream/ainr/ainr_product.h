/* AINR product binding, for the product build and this directory's tests. A product ships ONE model, DFN2
 * full or lite (the DFN2_MASK_ONLY variant), at one quantization: the build-time AINR_PRODUCT_MODEL_ID that
 * ainr.c resolves. Bind another with -DAINR_PRODUCT_MODEL_ID=<an ID below>; the IDs are placeholders. */

#ifndef AINR_PRODUCT_H
#define AINR_PRODUCT_H

#define AINR_MODEL_ID_DFN2_FULL      1
#define AINR_MODEL_ID_DFN2_FULL_ALT1 10
#define AINR_MODEL_ID_DFN2_FULL_ALT2 11
#define AINR_MODEL_ID_DFN2_LITE      2
#define AINR_MODEL_ID_DFN2_LITE_ALT1 20
#define AINR_MODEL_ID_DFN2_LITE_ALT2 21

#ifndef AINR_PRODUCT_MODEL_ID
#ifdef DFN2_MASK_ONLY
#define AINR_PRODUCT_MODEL_ID AINR_MODEL_ID_DFN2_LITE
#else
#define AINR_PRODUCT_MODEL_ID AINR_MODEL_ID_DFN2_FULL
#endif
#endif

#endif /* AINR_PRODUCT_H */
