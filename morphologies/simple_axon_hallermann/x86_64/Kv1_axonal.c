/* Created by Language version: 6.2.0 */
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
 
#define nrn_init _nrn_init__Kv1
#define _nrn_initial _nrn_initial__Kv1
#define nrn_cur _nrn_cur__Kv1
#define _nrn_current _nrn_current__Kv1
#define nrn_jacob _nrn_jacob__Kv1
#define nrn_state _nrn_state__Kv1
#define _net_receive _net_receive__Kv1 
#define rates rates__Kv1 
#define states states__Kv1 
 
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
#define gbar _p[0]
#define ikv1 _p[1]
#define gkv1 _p[2]
#define n _p[3]
#define h1 _p[4]
#define h2 _p[5]
#define ik _p[6]
#define ek _p[7]
#define nalpha _p[8]
#define nbeta _p[9]
#define h1alpha _p[10]
#define h1beta _p[11]
#define h2alpha _p[12]
#define h2beta _p[13]
#define h3alpha _p[14]
#define h3beta _p[15]
#define tadj _p[16]
#define tadjh _p[17]
#define Dn _p[18]
#define Dh1 _p[19]
#define Dh2 _p[20]
#define _g _p[21]
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
 
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
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_h2betafkt(void);
 static void _hoc_h2alphafkt(void);
 static void _hoc_h1betafkt(void);
 static void _hoc_h1alphafkt(void);
 static void _hoc_nbetafkt(void);
 static void _hoc_nalphafkt(void);
 static void _hoc_rates(void);
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
 "setdata_Kv1", _hoc_setdata,
 "h2betafkt_Kv1", _hoc_h2betafkt,
 "h2alphafkt_Kv1", _hoc_h2alphafkt,
 "h1betafkt_Kv1", _hoc_h1betafkt,
 "h1alphafkt_Kv1", _hoc_h1alphafkt,
 "nbetafkt_Kv1", _hoc_nbetafkt,
 "nalphafkt_Kv1", _hoc_nalphafkt,
 "rates_Kv1", _hoc_rates,
 0, 0
};
#define h2betafkt h2betafkt_Kv1
#define h2alphafkt h2alphafkt_Kv1
#define h1betafkt h1betafkt_Kv1
#define h1alphafkt h1alphafkt_Kv1
#define nbetafkt nbetafkt_Kv1
#define nalphafkt nalphafkt_Kv1
 extern double h2betafkt( double );
 extern double h2alphafkt( double );
 extern double h1betafkt( double );
 extern double h1alphafkt( double );
 extern double nbetafkt( double );
 extern double nalphafkt( double );
 /* declare global and static user variables */
#define h3tau h3tau_Kv1
 double h3tau = 0;
#define h3inf h3inf_Kv1
 double h3inf = 0;
#define h2tau h2tau_Kv1
 double h2tau = 0;
#define h2inf h2inf_Kv1
 double h2inf = 0;
#define h1tau h1tau_Kv1
 double h1tau = 0;
#define h1inf h1inf_Kv1
 double h1inf = 0;
#define ntau ntau_Kv1
 double ntau = 0;
#define ninf ninf_Kv1
 double ninf = 0;
#define q10h q10h_Kv1
 double q10h = 3;
#define q10 q10_Kv1
 double q10 = 3;
#define temp temp_Kv1
 double temp = 33;
#define vShift_inact vShift_inact_Kv1
 double vShift_inact = 0;
#define vShift vShift_Kv1
 double vShift = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "temp_Kv1", "degC",
 "ntau_Kv1", "ms",
 "h1tau_Kv1", "ms",
 "h2tau_Kv1", "ms",
 "h3tau_Kv1", "ms",
 "gbar_Kv1", "pS/um2",
 "ikv1_Kv1", "mA/cm2",
 "gkv1_Kv1", "mho/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h20 = 0;
 static double h10 = 0;
 static double n0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "vShift_Kv1", &vShift_Kv1,
 "vShift_inact_Kv1", &vShift_inact_Kv1,
 "temp_Kv1", &temp_Kv1,
 "q10_Kv1", &q10_Kv1,
 "q10h_Kv1", &q10h_Kv1,
 "ninf_Kv1", &ninf_Kv1,
 "ntau_Kv1", &ntau_Kv1,
 "h1inf_Kv1", &h1inf_Kv1,
 "h1tau_Kv1", &h1tau_Kv1,
 "h2inf_Kv1", &h2inf_Kv1,
 "h2tau_Kv1", &h2tau_Kv1,
 "h3inf_Kv1", &h3inf_Kv1,
 "h3tau_Kv1", &h3tau_Kv1,
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
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "6.2.0",
"Kv1",
 "gbar_Kv1",
 0,
 "ikv1_Kv1",
 "gkv1_Kv1",
 0,
 "n_Kv1",
 "h1_Kv1",
 "h2_Kv1",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 22, _prop);
 	/*initialize range parameters*/
 	gbar = 0;
 	_prop->param = _p;
 	_prop->param_size = 22;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Kv1_axonal_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
  hoc_register_prop_size(_mechtype, 22, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Kv1 /media/erebus/oslo/code/darpa/LFPy/examples/simple_axon_hallermann/x86_64/Kv1_axonal.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double a1 = 48.02279;
 static double a2 = 42.662;
 static double b1 = 58.44509;
 static double c1 = 13.44182;
 static double c2 = 49.38797;
 static double a1H1 = 1.300293e-03;
 static double a2H1 = 6.247311;
 static double b2H1 = 66.13843;
 static double c1H1 = 8.611435;
 static double c2H1 = 7.948768;
 static double a12H2fact = 1.554639e-02;
 static double propH2 = 0.73157;
 static double sToMs = 0.001;
static int _reset;
static char *modelname = "Axonal voltage-gated potassium current";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[3], _dlist1[3];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
   Dn = ( ninf - n ) / ntau ;
   Dh1 = ( h1inf - h1 ) / h1tau ;
   Dh2 = ( h2inf - h2 ) / h2tau ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rates ( _threadargscomma_ v ) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / ntau )) ;
 Dh1 = Dh1  / (1. - dt*( ( ( ( - 1.0 ) ) ) / h1tau )) ;
 Dh2 = Dh2  / (1. - dt*( ( ( ( - 1.0 ) ) ) / h2tau )) ;
 return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   rates ( _threadargscomma_ v ) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / ntau)))*(- ( ( ( ninf ) ) / ntau ) / ( ( ( ( - 1.0 ) ) ) / ntau ) - n) ;
    h1 = h1 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / h1tau)))*(- ( ( ( h1inf ) ) / h1tau ) / ( ( ( ( - 1.0 ) ) ) / h1tau ) - h1) ;
    h2 = h2 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / h2tau)))*(- ( ( ( h2inf ) ) / h2tau ) / ( ( ( ( - 1.0 ) ) ) / h2tau ) - h2) ;
   }
  return 0;
}
 
static int  rates (  double _lv ) {
   tadj = pow( q10 , ( ( celsius - temp ) / 10.0 ) ) ;
   tadjh = pow( q10h , ( ( celsius - temp ) / 10.0 ) ) ;
   nalpha = tadj * nalphafkt ( _threadargscomma_ _lv - vShift ) ;
   nbeta = tadj * nbetafkt ( _threadargscomma_ _lv - vShift ) ;
   ninf = nalpha / ( nalpha + nbeta ) ;
   ntau = 1.0 / ( nalpha + nbeta ) ;
   h1alpha = tadjh * h1alphafkt ( _threadargscomma_ _lv - vShift - vShift_inact ) ;
   h1beta = tadjh * h1betafkt ( _threadargscomma_ _lv - vShift - vShift_inact ) ;
   h1inf = h1alpha / ( h1alpha + h1beta ) ;
   h1tau = 1.0 / ( h1alpha + h1beta ) ;
   h2alpha = tadjh * h2alphafkt ( _threadargscomma_ _lv - vShift - vShift_inact ) ;
   h2beta = tadjh * h2betafkt ( _threadargscomma_ _lv - vShift - vShift_inact ) ;
   h2inf = h2alpha / ( h2alpha + h2beta ) ;
   h2tau = 1.0 / ( h2alpha + h2beta ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double nalphafkt (  double _lv ) {
   double _lnalphafkt;
 _lnalphafkt = sToMs * a1 * ( - ( _lv + b1 ) ) / ( exp ( - ( _lv + b1 ) / c1 ) - 1.0 ) ;
   
return _lnalphafkt;
 }
 
static void _hoc_nalphafkt(void) {
  double _r;
   _r =  nalphafkt (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double nbetafkt (  double _lv ) {
   double _lnbetafkt;
 _lnbetafkt = sToMs * a2 * exp ( - ( _lv ) / c2 ) ;
   
return _lnbetafkt;
 }
 
static void _hoc_nbetafkt(void) {
  double _r;
   _r =  nbetafkt (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double h1alphafkt (  double _lv ) {
   double _lh1alphafkt;
 _lh1alphafkt = sToMs * a1H1 * exp ( - ( _lv ) / c1H1 ) ;
   
return _lh1alphafkt;
 }
 
static void _hoc_h1alphafkt(void) {
  double _r;
   _r =  h1alphafkt (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double h1betafkt (  double _lv ) {
   double _lh1betafkt;
 _lh1betafkt = sToMs * a2H1 / ( exp ( - ( _lv + b2H1 ) / c2H1 ) + 1.0 ) ;
   
return _lh1betafkt;
 }
 
static void _hoc_h1betafkt(void) {
  double _r;
   _r =  h1betafkt (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double h2alphafkt (  double _lv ) {
   double _lh2alphafkt;
 _lh2alphafkt = sToMs * a12H2fact * a1H1 * exp ( - ( _lv ) / c1H1 ) ;
   
return _lh2alphafkt;
 }
 
static void _hoc_h2alphafkt(void) {
  double _r;
   _r =  h2alphafkt (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double h2betafkt (  double _lv ) {
   double _lh2betafkt;
 _lh2betafkt = sToMs * a12H2fact * a2H1 / ( exp ( - ( _lv + b2H1 ) / c2H1 ) + 1.0 ) ;
   
return _lh2betafkt;
 }
 
static void _hoc_h2betafkt(void) {
  double _r;
   _r =  h2betafkt (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  h2 = h20;
  h1 = h10;
  n = n0;
 {
   rates ( _threadargscomma_ v ) ;
   n = ninf ;
   h1 = h1inf ;
   h2 = h2inf ;
   }
  _sav_indep = t; t = _save;

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
  ek = _ion_ek;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   gkv1 = gbar * ( ( 1.0 - propH2 ) * pow( n , 8.0 ) * h1 + propH2 * pow( n , 8.0 ) * h2 ) ;
   ikv1 = ( 1e-4 ) * gkv1 * ( v - ek ) ;
   ik = ikv1 ;
   }
 _current += ik;

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
  ek = _ion_ek;
 _g = _nrn_current(_v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
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
  ek = _ion_ek;
 { error =  states();
 if(error){fprintf(stderr,"at line 112 in file Kv1_axonal.mod:\n	SOLVE states METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(n) - _p;  _dlist1[0] = &(Dn) - _p;
 _slist1[1] = &(h1) - _p;  _dlist1[1] = &(Dh1) - _p;
 _slist1[2] = &(h2) - _p;  _dlist1[2] = &(Dh2) - _p;
_first = 0;
}
