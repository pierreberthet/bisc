/* Created by Language version: 7.5.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__charge_
#define _nrn_initial _nrn_initial__charge_
#define nrn_cur _nrn_cur__charge_
#define _nrn_current _nrn_current__charge_
#define nrn_jacob _nrn_jacob__charge_
#define nrn_state _nrn_state__charge_
#define _net_receive _net_receive__charge_ 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define vmin _p[0]
#define tmin _p[1]
#define vmax _p[2]
#define tmax _p[3]
#define na_ch _p[4]
#define na_ch_overl _p[5]
#define overl _p[6]
#define na_ch_excess_ratio _p[7]
#define na_ch_before_peak _p[8]
#define na_ch_after_peak _p[9]
#define peak_reached _p[10]
#define peak_time _p[11]
#define na_ch_overl_tmp _p[12]
#define ina _p[13]
#define ik _p[14]
#define _g _p[15]
#define _ion_ina	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 /* declaration of user functions */
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_charge_", _hoc_setdata,
 0, 0
};
 /* declare global and static user variables */
#define peak_lowest peak_lowest_charge_
 double peak_lowest = 0;
#define peak_tolerance peak_tolerance_charge_
 double peak_tolerance = 0;
#define tEnd tEnd_charge_
 double tEnd = 0;
#define tStart tStart_charge_
 double tStart = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tStart_charge_", "ms",
 "tEnd_charge_", "ms",
 "peak_tolerance_charge_", "mV",
 "peak_lowest_charge_", "mV",
 "vmin_charge_", "millivolt",
 "tmin_charge_", "ms",
 "vmax_charge_", "millivolt",
 "tmax_charge_", "ms",
 "na_ch_charge_", "milliamp/cm2",
 "na_ch_overl_charge_", "milliamp/cm2",
 0,0
};
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "tStart_charge_", &tStart_charge_,
 "tEnd_charge_", &tEnd_charge_,
 "peak_tolerance_charge_", &peak_tolerance_charge_,
 "peak_lowest_charge_", &peak_lowest_charge_,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.5.0",
"charge_",
 0,
 "vmin_charge_",
 "tmin_charge_",
 "vmax_charge_",
 "tmax_charge_",
 "na_ch_charge_",
 "na_ch_overl_charge_",
 "overl_charge_",
 "na_ch_excess_ratio_charge_",
 "na_ch_before_peak_charge_",
 "na_ch_after_peak_charge_",
 "peak_reached_charge_",
 "peak_time_charge_",
 0,
 0,
 0};
 static Symbol* _na_sym;
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 16, _prop);
 	/*initialize range parameters*/
 	_prop->param = _p;
 	_prop->param_size = 16;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 2, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 	_ppvar[0]._pval = &prop_ion->param[3]; /* ina */
 prop_ion = need_memb(_k_sym);
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 
}
 static void _initlists();
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _charge_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("na", -10000.);
 	ion_reg("k", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
  hoc_register_prop_size(_mechtype, 16, 2);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 charge_ /nird/home/u1/berthetp/darpa/bisc/morphologies/HallermannEtAl2012/x86_64/charge.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "calculates Na+/K+ charge overlap and excess Na+ influx ";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
{
 {
   vmin = 1e6 ;
   tmin = 0.0 ;
   vmax = - 1e6 ;
   tmax = 0.0 ;
   peak_reached = 0.0 ;
   peak_time = 0.0 ;
   na_ch = 0.0 ;
   na_ch_before_peak = 0.0 ;
   na_ch_after_peak = 0.0 ;
   na_ch_excess_ratio = 0.0 ;
   na_ch_overl = 0.0 ;
   overl = 0.0 ;
   }

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ina = _ion_ina;
  ik = _ion_ik;
 initmodel();
}}

static double _nrn_current(double _v){double _current=0.;v=_v;{
} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ina = _ion_ina;
  ik = _ion_ik;
 {
   
/*VERBATIM*/
      if (t > tStart) {
		if (t < tEnd) {
			if (v < vmin) {
        		        vmin = v;
                		tmin = t;
		        }
		        if (v > vmax) {
        		        vmax = v;
	                	tmax = t;
		        }
 			na_ch = na_ch + ina;
			na_ch_overl_tmp = ina;				
			if (-ik > ina) {
                		na_ch_overl_tmp = -ik;
		        }
			na_ch_overl = na_ch_overl + na_ch_overl_tmp;
			if (na_ch !=  0) {	//na_ch is negative
                		overl = (na_ch - na_ch_overl) / na_ch;
			}
			if ( (v < vmax - peak_tolerance) && (v > peak_lowest) && (peak_reached == 0) ) {
				peak_reached = 1;
				peak_time = t;
			}
			if (peak_reached == 0) {
				na_ch_before_peak = na_ch_before_peak + ina;			
			} else {
				na_ch_after_peak = na_ch_after_peak + ina;
			}
			if (na_ch_before_peak != 0) {
				na_ch_excess_ratio = (na_ch_before_peak + na_ch_after_peak) / na_ch_before_peak;
			}
		}
	}
 }
}}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}
