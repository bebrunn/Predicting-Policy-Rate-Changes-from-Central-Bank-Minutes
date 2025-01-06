clear

cd "C:\Users\bbrunner\Desktop\thesis_export"
use master_brunner_data.dta

tsset index

*** Baseline model and extension
** Ordered probit
oprobit PolChange_t1 PolChange_t NetHawk_stan, vce(cluster index)
estimates store probit1

oprobit PolChange_t1 PolChange_t NetHawk_stan dInf, vce(cluster index)
estimates store probit2

oprobit PolChange_t1 PolChange_t NetHawk_stan dY, vce(cluster index)
estimates store probit3
 
oprobit PolChange_t1 PolChange_t NetHawk_stan CZK, vce(cluster index)
estimates store probit4

oprobit PolChange_t1 PolChange_t NetHawk_stan dInf dY CZK, vce(cluster index)
estimates store probit5

* Compute marginal effects
margins, dydx(*) atmeans

esttab probit1 probit2 probit3 probit4 probit5 using probit1.rtf, ///
    title("Table 1: Ordered Probit of Policy Change on Sentiment Index") ///
    se star(* 0.10 ** 0.05 *** 0.01) label ///
    mtitle("" "" "" "" "") ///
    stats(N r2_p)
	
*** Robustness checks
** RoBERTa sentiment dummies
encode sentiment, gen(sentiment_num)
tabulate sentiment_num, gen(sentiment_dummy)

oprobit PolChange_t1 PolChange_t sentiment_dummy1 sentiment_dummy2, vce(cluster index)
estimates store dum_probit1

oprobit PolChange_t1 PolChange_t sentiment_dummy1 sentiment_dummy2 dInf, vce(cluster index)
estimates store dum_probit2

oprobit PolChange_t1 PolChange_t sentiment_dummy1 sentiment_dummy2 dY, vce(cluster index)
estimates store dum_probit3
 
oprobit PolChange_t1 PolChange_t sentiment_dummy1 sentiment_dummy2 CZK, vce(cluster index)
estimates store dum_probit4

oprobit PolChange_t1 PolChange_t sentiment_dummy1 sentiment_dummy2 dInf dY CZK, vce(cluster index)
estimates store dum_probit5

* Compute marginal effects
margins, dydx(*) atmeans

esttab dum_probit1 dum_probit2 dum_probit3 dum_probit4 dum_probit5 using dummy_probit1.rtf, ///
    title("Table 1: Ordered Probit of Policy Change on Sentiment Index") ///
    se star(* 0.10 ** 0.05 *** 0.01) label ///
    mtitle("" "" "" "" "") ///
    stats(N r2_p)
	
** BERT net-hawkishness index
oprobit PolChange_t1 PolChange_t NetHawk_bert_stan, vce(cluster index)
estimates store bert_probit1

oprobit PolChange_t1 PolChange_t NetHawk_bert_stan dInf, vce(cluster index)
estimates store bert_probit2

oprobit PolChange_t1 PolChange_t NetHawk_bert_stan dY, vce(cluster index)
estimates store bert_probit3
 
oprobit PolChange_t1 PolChange_t NetHawk_bert_stan CZK, vce(cluster index)
estimates store bert_probit4

oprobit PolChange_t1 PolChange_t NetHawk_bert_stan dInf dY CZK, vce(cluster index)
estimates store bert_probit5

* Compute marginal effects
margins, dydx(*) atmeans

esttab bert_probit1 bert_probit2 bert_probit3 bert_probit4 bert_probit5 using bert_probit1.rtf, ///
    title("Table 1: Ordered Probit of Policy Change on Sentiment Index") ///
    se star(* 0.10 ** 0.05 *** 0.01) label ///
    mtitle("" "" "" "" "") ///
    stats(N r2_p)

	
*** Heterogeneity across time
** Ordered Probit
* First period (1998/01-2006/12): 0-115
oprobit PolChange_t1 PolChange_t NetHawk_stan if inrange(index, 0, 115), vce(cluster index)
estimates store probit1_0_115

oprobit PolChange_t1 PolChange_t NetHawk_stan dInf dY CZK if inrange(index, 0, 115), vce(cluster index)
estimates store probit2_0_115

margins, dydx(*) atmeans

* Second period (2007/01 - 2016/12): 116-199
oprobit PolChange_t1 PolChange_t NetHawk_stan if inrange(index, 116, 199), vce(cluster index)
estimates store probit1_116_199

oprobit PolChange_t1 PolChange_t NetHawk_stan dInf dY CZK if inrange(index, 116, 199), vce(cluster index)
estimates store probit2_116_199

margins, dydx(*) atmeans


* Third period (2017/08-2024/09): 200-264
oprobit PolChange_t1 PolChange_t NetHawk_stan if inrange(index, 200, 264), vce(cluster index)
estimates store probit1_200_264

oprobit PolChange_t1 PolChange_t NetHawk_stan dInf dY CZK if inrange(index, 200, 264), vce(cluster index)
estimates store probit2_200_264

margins, dydx(*) atmeans


* Export results to a table
esttab probit1_0_115 probit2_0_115 probit1_116_199 probit2_116_199 probit1_200_264 probit2_200_264 using probit_time_het.rtf, ///
    title("Table 1: Ordered Probit of Policy Change on Sentiment Index across Time Periods") ///
    se star(* 0.10 ** 0.05 *** 0.01) label ///
    mtitle("1998-2006" "1998-2006" "2007-2016" "2007-2016" "2017-2024" "2017-2024") ///
    stats(N r2_p)

