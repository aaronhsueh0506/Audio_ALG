/* AINR product binding, for the product build and this directory's tests. A product ships ONE model, DFN2
 * full or lite (the DFN2_MASK_ONLY variant): the build-time AINR_PRODUCT_MODEL_ID that ainr.c resolves. Bind another
 * with -DAINR_PRODUCT_MODEL_ID=<an ID below>; the IDs are placeholders. AINR_MODEL_ID_DFN2_FULL_L0 is the full model
 * with mask_lookahead 0 at every grid. The quantization is the sample's choice of model directory. */

#ifndef AINR_PRODUCT_H
#define AINR_PRODUCT_H

#define AINR_MODEL_ID_DFN2_FULL      1
#define AINR_MODEL_ID_DFN2_FULL_L0   3
#define AINR_MODEL_ID_DFN2_LITE      2

#ifndef AINR_PRODUCT_MODEL_ID
#ifdef DFN2_MASK_ONLY
#define AINR_PRODUCT_MODEL_ID AINR_MODEL_ID_DFN2_LITE
#else
#define AINR_PRODUCT_MODEL_ID AINR_MODEL_ID_DFN2_FULL
#endif
#endif

#endif /* AINR_PRODUCT_H */
