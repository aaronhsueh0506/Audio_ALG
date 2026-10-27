/* Shared sample-driver control flow: CLI boundaries, failures and cleanup. */
#include "mcab_sample_driver.h"
#include "common/example_cli.h"
#include <assert.h>
#ifdef NDEBUG
#error "test_mcab_sample_driver requires assertions"
#endif
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct Probe {
    int parse_rc,setup_rc,read_rc,process_rc,consume_rc;
    int cleaned,processed,consumed,finished,nonfinite,frames,finite,completed;
    float output[2];
} Probe;
static int parse(void *u,int argc,char **argv) {
    (void)argc;(void)argv;return ((Probe*)u)->parse_rc;
}
static int setup(void *u,McabSampleHopView *v) {
    Probe *p=u;v->hop=2;v->output=p->output;v->check_finite=p->finite;
    return p->setup_rc;
}
static int read_hop(void *u,McabSampleHopView *v,int frame) {
    Probe *p=u;(void)v;return frame < 3 ? p->read_rc : 0;
}
static int process(void *u,McabSampleHopView *v,int frame) {
    Probe *p=u;(void)v;(void)frame;p->processed++;return p->process_rc;
}
static int consume(void *u,McabSampleHopView *v,int frame) {
    Probe *p=u;(void)v;(void)frame;p->consumed++;return p->consume_rc;
}
static void nonfinite(void *u) {((Probe*)u)->nonfinite++;}
static int finish(void *u,McabSampleHopView *v,int frames) {
    Probe *p=u;(void)v;p->finished++;p->frames=frames;return 0;
}
static void cleanup(void *u) {((Probe*)u)->cleaned++;}
static void complete(void *u) { Probe *p=u;assert(p->cleaned==1);p->completed++; }
static int run(Probe *p) {
    static const McabSampleOps ops={parse,2,setup,read_hop,process,consume,
                                   nonfinite,finish,cleanup,complete};
    char *args[]={"probe",NULL};return mcab_sample_main(1,args,&ops,p);
}
static Probe fresh(void) {Probe p={0};p.read_rc=1;p.finite=1;return p;}
static int set(void *u,const char *name,const char *value) {
    int *seen=u;
    if (!strcmp(name,"--help")) return 1;
    assert(value && !strcmp(value,"12"));(*seen)++;return 0;
}
static int unknown(void *u,const char *name) {(void)u;(void)name;return 2;}
static int fake_query(const McabConfig *c,const McabSampleBeamParams *b,McabMemReq *r,
                      unsigned long long *total) {
    (void)c;(void)b;memset(r,0,sizeof(*r));r->bytes=100;r->alignment=16;
    r->backend_id=MCAB_BACKEND_NE10;*total=120;return 0;
}
static int refusing_query(const McabConfig *c,const McabSampleBeamParams *b,McabMemReq *r,
                          unsigned long long *total) {
    (void)c;(void)b;(void)r;(void)total;return -1;
}
int main(void) {
    Probe p=fresh();assert(run(&p)==0);assert(p.processed==3 && p.consumed==3);
    assert(p.frames==3 && p.finished==1 && p.cleaned==1 && p.completed==1);
    p=fresh();p.parse_rc=1;assert(run(&p)==0 && p.cleaned==0);
    p=fresh();p.parse_rc=2;assert(run(&p)==2 && p.cleaned==0);
    p=fresh();p.setup_rc=1;assert(run(&p)==1 && p.cleaned==1 && p.processed==0);
    p=fresh();p.setup_rc=2;assert(run(&p)==0 && p.cleaned==1 && p.finished==0);
    p=fresh();p.read_rc=0;assert(run(&p)==0 && p.frames==0 && p.cleaned==1);
    p=fresh();p.read_rc=-1;assert(run(&p)==1 && p.finished==0 && p.cleaned==1);
    p=fresh();p.process_rc=1;assert(run(&p)==1 && p.processed==1 && p.cleaned==1 && p.consumed==0);
    p=fresh();p.consume_rc=1;assert(run(&p)==1 && p.consumed==1 && p.cleaned==1 && p.finished==0);
    p=fresh();p.output[1]=NAN;assert(run(&p)==1 && p.nonfinite==1 && p.cleaned==1 && p.consumed==0);
    p=fresh();p.output[1]=NAN;p.finite=0;assert(run(&p)==0 && p.nonfinite==0 && p.consumed==3);
    {   /* finish is optional: NULL means success after the hop loop. */
        static const McabSampleOps no_finish={parse,2,setup,read_hop,process,
                                              NULL,NULL,NULL,cleanup,complete};
        char *args[]={"probe",NULL};
        p=fresh();assert(mcab_sample_main(1,args,&no_finish,&p)==0);
        assert(p.processed==3 && p.cleaned==1 && p.completed==1);
    }
    const McabSampleCliOption opts[]={{"--value",1},{"--help",0}};
    int seen=0;char *ok[]={"probe","--value","12"};
    assert(mcab_sample_scan_cli(3,ok,opts,2,&seen,set,unknown)==0 && seen==1);
    char *missing[]={"probe","--value"};
    assert(mcab_sample_scan_cli(2,missing,opts,2,&seen,set,unknown)==2);
    char *help[]={"probe","--help","--bogus"};
    assert(mcab_sample_scan_cli(3,help,opts,2,&seen,set,unknown)==1);
    McabSampleSpatialOptions o;char *legacy[]={"probe","--sample-rate","16000Hz"};
    assert(mcab_sample_parse_spatial_cli(3,legacy,0,&o)==0 && o.sample_rate==16000);
    assert(mcab_sample_parse_spatial_cli(3,legacy,1,&o)==2);
    {   /* Strict numbers: empty, padded, trailing garbage, outside int, non-finite. */
        static const char *bad_int[]={"", " 3", "3 ", "x3", "3x", "+-1", "2147483648", "-2147483649",
                                      "99999999999999999999999"};
        static const char *bad_float[]={"", " 1", "1 ", "1dB", "inf", "nan", "1e99"};
        int iv=7;float fv=7.0f;size_t k;
        for(k=0;k<sizeof(bad_int)/sizeof(bad_int[0]);++k) assert(example_cli_parse_int(bad_int[k],&iv)==-1 && iv==7);
        for(k=0;k<sizeof(bad_float)/sizeof(bad_float[0]);++k) assert(example_cli_parse_float(bad_float[k],&fv)==-1 && fv==7.0f);
        assert(example_cli_parse_int("-2147483648",&iv)==0 && iv==-2147483647-1);
        assert(example_cli_parse_int("+3",&iv)==0 && iv==3);
        assert(example_cli_parse_float("-0.5",&fv)==0 && fv==-0.5f);
        char *empty_doa[]={"probe","--print-mem-size","--sample-rate","16000","--fixed-doa-deg",""};
        assert(mcab_sample_parse_spatial_cli(6,empty_doa,1,&o)==2);
        assert(mcab_sample_parse_spatial_cli(6,empty_doa,0,&o)==0);
    }
    ExampleCliDelayOptions d={AEC_DELAY_FIXED,1,0,0,1};
    assert(example_cli_delay_validate(&d,"probe")==0 && d.num_filters==DA_NUM_FILTERS);
    {   /* The budget printer: the unit's size comes from the caller's query. */
        McabConfig cfg;
        McabSampleBeamParams bp;
        McabSampleMemLine extra[2]={{"Chain:",28},{"Stage:",4}};
        memset(&cfg,0,sizeof(cfg));memset(&bp,0,sizeof(bp));cfg.sample_rate=16000;
        assert(mcab_sample_print_mem_budget("probe",&cfg,&bp,fake_query,extra,2)==0);
        assert(mcab_sample_print_mem_budget("probe",&cfg,&bp,fake_query,NULL,0)==0);
        assert(mcab_sample_print_mem_budget("probe",&cfg,&bp,refusing_query,extra,2)==-1);
    }
    puts("mcab_sample_driver: control-flow and CLI gates PASS");return 0;
}
