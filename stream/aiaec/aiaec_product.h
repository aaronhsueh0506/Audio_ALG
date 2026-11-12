/* Product binding for the product build and this directory's tests, not customer code: one model per product, chosen by
 * AIAEC_PRODUCT_MODEL_ID (default: the one model the build enables, DeepVQE-S when both). An unknown ID or a model the build
 * lacks is refused with AIAEC_ERR_MODEL_ID by every entry. */

#ifndef AIAEC_PRODUCT_H
#define AIAEC_PRODUCT_H

#ifndef AIAEC_ENABLE_DEEPVQE_S
#define AIAEC_ENABLE_DEEPVQE_S 1
#endif
#ifndef AIAEC_ENABLE_ALIGN_ULCNET
#define AIAEC_ENABLE_ALIGN_ULCNET 1
#endif

#define AIAEC_MODEL_ID_DEEPVQE_S     1
#define AIAEC_MODEL_ID_ALIGN_ULCNET  10

#ifndef AIAEC_PRODUCT_MODEL_ID
#if AIAEC_ENABLE_ALIGN_ULCNET && !AIAEC_ENABLE_DEEPVQE_S
#define AIAEC_PRODUCT_MODEL_ID AIAEC_MODEL_ID_ALIGN_ULCNET
#else
#define AIAEC_PRODUCT_MODEL_ID AIAEC_MODEL_ID_DEEPVQE_S
#endif
#endif

#endif /* AIAEC_PRODUCT_H */
