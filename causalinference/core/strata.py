import causalinference.utils.tools as tools


class Strata(object):

	"""
	List-like object containing the stratified propensity bins.
	"""

	def __init__(self, strata, subsets, pscore):

		self._strata = strata
		for stratum, subset in zip(self._strata, subsets):
			pscore_sub = pscore[subset]
			stratum.raw_data._dict['pscore'] = pscore_sub
			D_sub = stratum.raw_data['D']
			pscore_sub_c = pscore_sub[D_sub==0]
			pscore_sub_t = pscore_sub[D_sub==1]
			stratum.summary_stats._summarize_pscore(pscore_sub_c,
			                                        pscore_sub_t)
			

	def __len__(self):

		return len(self._strata)


	def __getitem__(self, index):

		return self._strata[index]


	def __str__(self):

		table_width = 80

		output = '\n'
		output += 'Stratification Summary\n\n'

		entries1 = ['', 'Propensity Score', 'Sample Size',
		            'Ave. Propensity', 'Outcome']
		entry_types1 = ['string']*5
		col_spans1 = [1, 2, 2, 2, 1]
		output += tools.add_row(entries1, entry_types1,
		                        col_spans1, table_width)

		entries2 = ['Stratum', 'Min.', 'Max.', 'Controls', 'Treated',
		            'Controls', 'Treated', 'Raw-diff']
		entry_types2 = ['string']*8
		col_spans2 = [1]*8
		output += tools.add_row(entries2, entry_types2,
		                        col_spans2, table_width)
		output += tools.add_line(table_width)

		strata = self._strata
		entry_types3 = ['integer', 'float', 'float', 'integer',
		                'integer', 'float', 'float', 'float']
		for i in range(len(strata)):
			summary = strata[i].summary_stats
			N_c, N_t = summary['N_c'], summary['N_t']
			p_min, p_max = summary['p_min'], summary['p_max']
			p_c_mean = summary['p_c_mean']
			p_t_mean = summary['p_t_mean']
			within = summary['rdiff']
			entries3 = [i+1, p_min, p_max, N_c, N_t,
			            p_c_mean, p_t_mean, within]
			output += tools.add_row(entries3, entry_types3,
			                        col_spans2, table_width)

		return output

