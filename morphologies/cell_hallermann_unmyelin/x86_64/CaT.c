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
 
#define nrn_init _nrn_init__it2
#define _nrn_initial _nrn_initial__it2
#define nrn_cur _nrn_cur__it2
#define _nrn_current _nrn_current__it2
#define nrn_jacob _nrn_jacob__it2
#define nrn_state _nrn_state__it2
#define _net_receive _net_receive__it2 
#define _f_trates _f_trates__it2 
#define rates rates__it2 
#define states states__it2 
#define trates trates__it2 
 
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
#define gcabar _p[0]
#define icaT _p[1]
#define gcaT _p[2]
#define minf _p[3]
#define hinf _p[4]
#define mtau _p[5]
#define htau _p[6]
#define m _p[7]
#define h _p[8]
#define ica _p[9]
#define eca _p[10]
#define tadj _p[11]
#define Dm _p[12]
#define Dh _p[13]
#define _g _p[14]
#define _ion_eca	*_ppvar[0]._pval
#define _ion_ica	*_ppvar[1]._pval
#define _ion_dicadv	*_ppvar[2]._pval
 
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
 static void _hoc_rates(void);
 static void _hoc_states(void);
 static void _hoc_trates(void);
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
 "setdata_it2", _hoc_setdata,
 "rates_it2", _hoc_rates,
 "states_it2", _hoc_states,
 "trates_it2", _hoc_trates,
 0, 0
};
 /* declare global and static user variables */
#define ah ah_it2
 double ah = 85;
#define am am_it2
 double am = 3;
#define cai cai_it2
 double cai = 0;
#define cao cao_it2
 double cao = 2;
#define usetable usetable_it2
 double usetable = 1;
#define vh2 vh2_it2
 double vh2 = 405;
#define vh1 vh1_it2
 double vh1 = 46;
#define vm2 vm2_it2
 double vm2 = 100;
#define vm1 vm1_it2
 double vm1 = 25;
#define vwh vwh_it2
 double vwh = 5;
#define vwm vwm_it2
 double vwm = 7.4;
#define v12h v12h_it2
 double v12h = 78;
#define v12m v12m_it2
 double v12m = 50;
#define vmax vmax_it2
 double vmax = 100;
#define vmin vmin_it2
 double vmin = -120;
#define vshift vshift_it2
 double vshift = 0;
#define wh2 wh2_it2
 double wh2 = 50;
#define wh1 wh1_it2
 double wh1 = 4;
#define wm2 wm2_it2
 double wm2 = 15;
#define wm1 wm1_it2
 double wm1 = 20;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_it2", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "vshift_it2", "mV",
 "cao_it2", "mM",
 "cai_it2", "mM",
 "vmin_it2", "mV",
 "vmax_it2", "mV",
 "v12m_it2", "mV",
 "v12h_it2", "mV",
 "vwm_it2", "mV",
 "vwh_it2", "mV",
 "am_it2", "mV",
 "ah_it2", "mV",
 "vm1_it2", "mV",
 "vm2_it2", "mV",
 "vh1_it2", "mV",
 "vh2_it2", "mV",
 "wm1_it2", "mV",
 "wm2_it2", "mV",
 "wh1_it2", "mV",
 "wh2_it2", "mV",
 "gcabar_it2", "pS/um2",
 "icaT_it2", "mA/cm2",
 "gcaT_it2", "pS/um2",
 "mtau_it2", "ms",
 "htau_it2", "ms",
 0,0
};
 static double delta_t = 1;
 static double h0 = 0;
 static double m0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "vshift_it2", &vshift_it2,
 "cao_it2", &cao_it2,
 "cai_it2", &cai_it2,
 "vmin_it2", &vmin_it2,
 "vmax_it2", &vmax_it2,
 "v12m_it2", &v12m_it2,
 "v12h_it2", &v12h_it2,
 "vwm_it2", &vwm_it2,
 "vwh_it2", &vwh_it2,
 "am_it2", &am_it2,
 "ah_it2", &ah_it2,
 "vm1_it2", &vm1_it2,
 "vm2_it2", &vm2_it2,
 "vh1_it2", &vh1_it2,
 "vh2_it2", &vh2_it2,
 "wm1_it2", &wm1_it2,
 "wm2_it2", &wm2_it2,
 "wh1_it2", &wh1_it2,
 "wh2_it2", &wh2_it2,
 "usetable_it2", &usetable_it2,
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
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "6.2.0",
"it2",
 "gcabar_it2",
 0,
 "icaT_it2",
 "gcaT_it2",
 "minf_it2",
 "hinf_it2",
 "mtau_it2",
 "htau_it2",
 0,
 "m_it2",
 "h_it2",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 15, _prop);
 	/*initialize range parameters*/
 	gcabar = 8;
 	_prop->param = _p;
 	_prop->param_size = 15;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* eca */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _CaT_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
  hoc_register_prop_size(_mechtype, 15, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 it2 /media/erebus/oslo/code/darpa/LFPy/examples/cell_hallermann_unmyelin/x86_64/CaT.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 96485.3;
 static double R = 8.31342;
 static double PI = 3.14159;
 static double _zmexp , _zhexp ;
 static double *_t_minf;
 static double *_t__zmexp;
 static double *_t_hinf;
 static double *_t__zhexp;
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_trates(double);
static int rates(double);
static int states();
static int trates(double);
 static void _n_trates(double);
 
static int  states (  ) {
   trates ( _threadargscomma_ v + vshift ) ;
   m = m + _zmexp * ( minf - m ) ;
   h = h + _zhexp * ( hinf - h ) ;
   
/*VERBATIM*/
	return 0;
  return 0; }
 
static void _hoc_states(void) {
  double _r;
   _r = 1.;
 states (  );
 hoc_retpushx(_r);
}
 static double _mfac_trates, _tmin_trates;
 static void _check_trates();
 static void _check_trates() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_dt;
  if (!usetable) {return;}
  if (_sav_dt != dt) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_trates =  vmin ;
   _tmax =  vmax ;
   _dx = (_tmax - _tmin_trates)/199.; _mfac_trates = 1./_dx;
   for (_i=0, _x=_tmin_trates; _i < 200; _x += _dx, _i++) {
    _f_trates(_x);
    _t_minf[_i] = minf;
    _t__zmexp[_i] = _zmexp;
    _t_hinf[_i] = hinf;
    _t__zhexp[_i] = _zhexp;
   }
   _sav_dt = dt;
  }
 }

 static int trates(double _lv){ _check_trates();
 _n_trates(_lv);
 return 0;
 }

 static void _n_trates(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_trates(_lv); return; 
}
 _xi = _mfac_trates * (_lv - _tmin_trates);
 if (isnan(_xi)) {
  minf = _xi;
  _zmexp = _xi;
  hinf = _xi;
  _zhexp = _xi;
  return;
 }
 if (_xi <= 0.) {
 minf = _t_minf[0];
 _zmexp = _t__zmexp[0];
 hinf = _t_hinf[0];
 _zhexp = _t__zhexp[0];
 return; }
 if (_xi >= 199.) {
 minf = _t_minf[199];
 _zmexp = _t__zmexp[199];
 hinf = _t_hinf[199];
 _zhexp = _t__zhexp[199];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 minf = _t_minf[_i] + _theta*(_t_minf[_i+1] - _t_minf[_i]);
 _zmexp = _t__zmexp[_i] + _theta*(_t__zmexp[_i+1] - _t__zmexp[_i]);
 hinf = _t_hinf[_i] + _theta*(_t_hinf[_i+1] - _t_hinf[_i]);
 _zhexp = _t__zhexp[_i] + _theta*(_t__zhexp[_i+1] - _t__zhexp[_i]);
 }

 
static int  _f_trates (  double _lv ) {
   double _ltinc ;
 rates ( _threadargscomma_ _lv ) ;
   _ltinc = - dt ;
   _zmexp = 1.0 - exp ( _ltinc / mtau ) ;
   _zhexp = 1.0 - exp ( _ltinc / htau ) ;
    return 0; }
 
static void _hoc_trates(void) {
  double _r;
    _r = 1.;
 trates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int  rates (  double _lv_ ) {
   double _la , _lb ;
 minf = 1.0 / ( 1.0 + exp ( - ( _lv_ + v12m ) / vwm ) ) ;
   hinf = 1.0 / ( 1.0 + exp ( ( _lv_ + v12h ) / vwh ) ) ;
   mtau = ( am + 1.0 / ( exp ( ( _lv_ + vm1 ) / wm1 ) + exp ( - ( _lv_ + vm2 ) / wm2 ) ) ) ;
   htau = ( ah + 1.0 / ( exp ( ( _lv_ + vh1 ) / wh1 ) + exp ( - ( _lv_ + vh2 ) / wh2 ) ) ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ hoc_execerror("it2", "cannot be used with CVODE"); return 0;}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  h = h0;
  m = m0;
 {
   trates ( _threadargscomma_ v + vshift ) ;
   m = minf ;
   h = hinf ;
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
  eca = _ion_eca;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   gcaT = gcabar * m * m * h ;
   icaT = gcaT * ( 1e-4 ) * ( v - eca ) ;
   ica = icaT ;
   }
 _current += ica;

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
  eca = _ion_eca;
 _g = _nrn_current(_v + .001);
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_v);
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
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
  eca = _ion_eca;
 { error =  states();
 if(error){fprintf(stderr,"at line 82 in file CaT.mod:\n        SOLVE states\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
   _t_minf = makevector(200*sizeof(double));
   _t__zmexp = makevector(200*sizeof(double));
   _t_hinf = makevector(200*sizeof(double));
   _t__zhexp = makevector(200*sizeof(double));
_first = 0;
}
