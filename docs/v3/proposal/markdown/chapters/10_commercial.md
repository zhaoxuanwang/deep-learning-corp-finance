# Commercial Product {#sec:ch-commercial}

This chapter proposes a variety of commercial products developed from the project. Each section describes one product category: context, potential buyer, commercial value, example usage, and pricing strategy.

Across all product category, the core engine is a *quantitative model and system* consisting of:

- **Information input**: data, parameters, assumptions, theories, configurations;

- **Processing**: machine learning, statistical and econometric methods, numerical optimization;

- **Information output**: quantitative estimates and forecasts.

Importantly, this model does not directly make financial decisions or actions. It is used as a tool to support the user's judgment and decision-making.

## Product 1: Corporate Finance Advisory (CFA) {#sec:product-1-corporate-finance-advisory-cfa}

Corporate Finance Advisory (CFA) is a global, multi-disciplinary solutions team specializing in structured M&A and capital markets that is focused on differentiating J.P. Morgan's Investment Banking services. CFA works with firms in all industries and partners across the bank to advise on solutions to problems with multifaceted, complex and unique features.[^1]

The model typically takes the corporate client's financial data and other information as input, and produce a credible interval of the optimal capital structure and policies as key outputs. These outputs are further processed to analyze counterfactual scenarios and generate other enhanced insights.

### Buyer {#sec:buyer}

Potential buyer is JPMC's CFA team and similar teams in other banks. This product directly supports:

- **Corporate finance solutions**: Provide analysis and recommendations on capital structure, capital allocation and shareholder distribution policy.

- **Ratings advisory**: Strategic and tactical advice on management of rating agency relationships and optimization of capital structures for desired credit rating objectives.

- **Structuring and solutions**: Structure products and alternative financing solutions relating to strategic M&A and capital markets.

- **Enhanced insight generation**: Leverage new technologies to improve both data analytics and visualization across all product and coverage groups.

### Example {#sec:example}

A company holds \$2 billion in surplus cash and wants to return it to shareholders through a buyback, funded partly with new debt. Its board asks one question: does the plan raise value without threatening the company's credit rating? JPMorgan's CFA team advises on the answer. The standard approach relies on peer benchmarks, rating-agency ratio scorecards, stress scenarios, and experienced judgment. These can estimate how the company's ratios map to a likely rating, but they do not value the choice itself, weighing the gain in firm value against the added default risk in a unified, model-based framework.

The model answers the question directly. From the company's own financials, it simulates the buyback and measures two effects together: the change in firm value and the change in default risk. It shows that the full \$2 billion buyback raises value only modestly and lifts the chance of a one-notch downgrade to roughly 30%. A smaller \$1.5 billion buyback captures most of the value while keeping downgrade risk low. The team brings the board a clear tradeoff, grounded in the company's own numbers rather than peer averages.

### Value and Pricing {#sec:pricing}

CFA work is firm-specific advice on capital structure, allocation, payout, and ratings. The model produces exactly that counterfactual analysis quantitatively. For a banker, this is a differentiated pitch tool: "our proprietary model, estimated on your fundamentals, says X." This is also a way to standardize work that is currently bespoke and analyst-heavy.

- **Ceiling**: set by how much the bank believes the model lifts mandate wins and fees.

- **Floor**: set by the analyst hours it displaces.

**Realistic pricing strategy**: platform subscription plus per-user license (seats), with the value carried in the sales narrative rather than a success fee.

**Ideal pricing strategy**: causal attribution that pins down the success fee per deal (e.g., "our product uplifts 10% of the advisory fee"). We need to deploy causal inference tools for credible evaluation. For example, start with a randomized experiment that gives the tool to different teams internally, collect data on outcomes, and estimate the average treatment effects.

**Reason**: It is hard to quantify the product uplift of wins and fees. CFA is often a relationship and credibility product that supports the broader IB relationship, so part of the product's value is indirect and long-term.

## Product 2: Debt Capital Markets (DCM) {#sec:product-2-debt-capital-markets-dcm}

J.P. Morgan's Debt Capital Market team assists clients --- including corporations, financial institutions, private equity and strategic investors --- on a wide range of debt financing strategies.[^2]

### Buyer {#sec:buyer-1}

Potential buyer is JPMC's DCM team. This product directly supports:

- **Bond pricing advisory**: provide an independent fair-value anchor based on model estimates from the firm's fundamentals, which can be used to compare the market spread the bank derived from similar-rated bonds trade.

- **Liability management and structuring**: help the client decide whether to refinance, change the debt mix, or pick a tenor and seniority. Our model enables full dynamic counterfactual analysis across alternative capital structures.

### Value {#sec:value-1}

The bank captures this value indirectly. The client receives a better result than a peer-based rule would deliver, which strengthens the relationship and the bank's case to win the lead role, where several million dollars of underwriting fees are earned. Competing banks, working only from comparables, cannot produce the same fundamentals-based anchor. The tool does not replace the banker's judgment. It gives that judgment independent support.

### Example {#sec:example-1}

A company plans to raise \$1 billion by issuing a bond, and JPMorgan's DCM team must advise on the price. The standard method anchors the price to where comparable bonds trade, the issuer's own outstanding debt and similar-rated peers, plus a small new-issue concession. This reflects the market's current read and the issuer's rating, not an independent estimate built from the issuer's own fundamentals.

The model prices the bond from the issuer's fundamentals. It finds the company is stronger than its rating peers and supports a tighter spread. Carrying that independent anchor into the deal, the bank prices the bond about 15 basis points tighter than comparables alone would have set. On a \$1 billion ten-year bond, that saves the issuer roughly \$1.5 million a year, about \$15 million over its life.

### Pricing {#sec:pricing-1}

DCM decisions attach to basis points on large events (e.g., issuance, M&A). An independent fair-value anchor that helps price a deal a few bps better, or that strengthens the pitch to win the book-runner role, both has a clear and large per-deal value.

**Example**: A company wants to raise two billion dollars by issuing a 10-year bond. It hires banks to run the sale, and our tool is used by the bank on this deal. There are two places value can come from.

1.  **Winning the mandate**: If the tool's independent fair-value analysis makes the bank's pitch more convincing, the bank is more likely to be chosen as bookrunner. Suppose the bank's fee on a deal this size is a few million dollars.

2.  **Better pricing for the client**: Suppose the tool shows the bond can price 5 bps tighter than the comparable-bond view suggested. On two billion dollars, 5 bps is 0.05%, which is one million dollars a year that the borrower saves. This wins more future mandates and builds the bank's reputation.

**Pricing strategy**: a share of the value the bank can credibly attribute to the tool across a year of deals it touches, where that value is mostly extra mandate fees and proven cost savings.

## Product 3: Leveraged Finance and Private Credit {#sec:product-3-leveraged-finance-and-private-credit}

The model can be used as a fundamentals-based underwriting and monitoring tool for leveraged finance and private credit.

**Leveraged finance**: J.P. Morgan is the partner of choice for Corporate & Financial Sponsors across the globe with unmatched structuring, marketing and execution capabilities in Leveraged Finance. J.P. Morgan platform is the undisputed market leader in arranging Loans and underwriting bonds that go to fund event-driven financing transactions including Capital Expenditures, Acquisitions, Leveraged Buyouts and Dividend Recapitalizations.

**Investment grade finance**: J.P. Morgan works directly with issuers, including corporations and financial institutions seeking debt financing. J.P. Morgan advises clients on capital structure strategy, from regular-way bond refinancing or a simple bank loan to multi-billion-dollar capital raises across asset classes. Colleagues partner across capital markets products and country borders to deliver successful structuring, marketing, and pricing

**Private credit**: J.P. Morgan increases direct lending commitment to 50 billion USD. This strategic move is designed to extend the firm's direct lending capabilities and provide tailored private credit solutions to meet the evolving needs of clients.[^3]

### Buyer {#sec:buyer-2}

Internally it supports JPMorgan's Leveraged Finance group and its direct lending platform in CIB. Externally it is sellable to direct lenders, private credit funds, and leveraged buyout sponsors.

- **Underwriting and leverage sizing**: Give an independent, fundamentals-based estimate of how much debt a borrower can sustainably carry, and its default probability over the life of the loan, as a check on the deal team's comparable-based view before capital is committed.

- **Risk-based pricing**: Produce a fair credit spread for the proposed structure from the borrower's own fundamentals, flagging when a deal underpays the lender for the risk taken.

- **Scenario and covenant stress testing**: Run dynamic counterfactuals across leverage, tenor, and downturn cases, quantifying how default risk and covenant headroom (the leverage and coverage tests) move if earnings fall. This is timely, since 2026 is widely seen as the private credit market's first real stress test.

- **Portfolio monitoring and early warning**: Re-estimate the model each quarter across the existing loan book to flag borrowers whose default risk is rising before it surfaces in the financials, supporting risk governance and investor reporting.

### Value {#sec:value-2}

The larger value lies in the deals that our tool warns the fund to avoid. A single default on a loan this size can cost the fund \$80 million or more. Avoiding one such loss may outweigh many years of the subscription.

### Example {#sec:example-2}

A private credit fund must decide whether to lend \$200 million to a company that a private equity firm is buying. Its analysts compare the deal to how similar companies were financed and conclude the loan is safe and fairly priced. This peer comparison is the industry standard, and it has one structural weakness. No peer company is this borrower carrying this much new debt. The fund's existing tools, comparison tables and rating-style scorecards, all rest on peers and historical averages. None can model this specific borrower under this specific debt load.

Our structural model closes this gap. It uses data of the borrower's own financials and estimates the probability that the company cannot service the debt. It then computes the interest rate that fairly compensates the fund for that risk. In this case, our model confirms the loan is safe but flags the proposed rate as 50 basis points too low. The fund secures the higher rate. On a \$200 million loan, that adds \$1 million of income per year, about \$5 million over a five-year term.

## Model risk and deployment issues {#sec:model-risk-and-compliance}

The key issues for product deployment in real life are:

- Model validation, risk management and compliance.

- Both bankers and clients need to trust our model and estimates

- Ongoing monitoring after launch and performance evaluation.

The primary obstacle is **model risk management and compliance**: the product must address the related laws, rules, and supervisory guidance when deployed inside a global bank like JPMorgan. Specifically, the product needs to pass the requirements on model validation, risk management, AI-tool-use rules, and data protection policies.

The second issue is that **both bankers and clients need to trust our model and estimates**. It not only requires our model to be well tested and validated. The product should be tailored to the client's need and incorporate firm-specific information to make the estimates useful. A lot of work is needed to understand what the real needs of the client are, and how our model can be developed into a specialized tool to meet the need. This requires deeper industry knowledge, professional guidance, and collaboration with clients.

The third issue is **ongoing monitoring and improvement** after launch. Model design and estimates should be updated with new data and customer feedback. We need to develop credible metrics and methods to evaluate the effectiveness of our tool, and quantify its impact.

The plan considers two main jurisdictions: (1) the United States, where JPMorgan is supervised as a US bank, and (2) the Hong Kong SAR, where the bank operates through locally regulated entities.

**Scope of the product.** For classification purposes, the end product is defined as a quantitative/statistical model that consumes input data and configurations, and outputs statistical estimates. There are several important boundaries:

- The model is not a black box and is fully explainable.

- The model is not a generative model or an agentic system.

- AI-tool-use is restricted to coding implementation and testing, supervised and reviewed by human.

Most of the current AI-related obligations center on generative AI and agentic systems. Our model is designed such that it is not generative or agentic, and it is largely within the standard quantitative/statistical model category.

### Key principles of compliance {#sec:key-principles-of-compliance}

This section provides a high-level summary of the model risk governance and compliance guidelines in the United States and Hong Kong SAR.

- **A human stays responsible.** Final accountability rests with named people, not with the model. The model is decision support: a banker or officer reviews and signs off, and the model never makes a decision automatically on its own. (US: SR 26-2 governance; HK: HKMA Principle 1, SFC General Principle 9.)

- **No black box.** The model must be explainable to the people who rely on it and to independent reviewers, clearly enough that they can question and challenge it. We have to be able to say, in explicit terms, why the model produced a given number. (US: "effective challenge"; HK: HKMA Principle 3, "no black-box excuse.")

- **Reproducibility and record-keeping.** The model version, spec, parameters, inputs, and the output that informed any advice must be retained for the applicable period and reproducible on demand. Build versioning and an audit log into the product from the start. Baseline validation should emphasize reproducibility control.

- **Independent validation before it goes live.** The model must be tested and approved before it is used, and the builder cannot also be the validator. (US: SR 26-2 conceptual soundness and effective challenge; HK: HKMA Principle 5, SFC Internal Control Guidelines.)

- **Concrete and complete documentation.** The design, assumptions, model choices, data sources, and test results must be documented well enough for a third party to follow and reproduce the work. In practice, if it is not written down, it does not count as done. (US: SR 26-2 conceptual-soundness documentation; HK: across the AI principles and the conduct code.)

- **Out-of-sample testing.** The model must be checked against actual outcomes on data it was not built on (out-of-sample back-testing), not merely shown to fit its own training data. (US: SR 26-2 outcomes analysis; HK: SPM CA-G-3, validation "should not be limited to back-testing.")

- **Monitoring after launch.** Performance must be monitored over time, and the model re-checked or re-estimated when markets or conditions change. (US: SR 26-2 ongoing monitoring; HK: HKMA AI principles.)

- **Be honest about limitations.** Known weaknesses must be disclosed to users, and use restricted where the model is weak. For us this means reporting ranges rather than false precision where parameters are only weakly identified. (US: SR 26-2 on use limitations; HK: same expectation.)

- **Input data quality.** The data feeding the model must be appropriate and of good quality, and the data choices documented. (US: SR 26-2 data selection; HK: HKMA Principle 4, data quality.)

- **If sold, give the buyer enough to validate it.** A buyer institution stays accountable for any model it uses, so we must ship a transparency package that lets its own reviewers understand and check the model without us handing over proprietary code. (US: SR 26-2, Section VII; HK: third-party provider principles.)

[^1]: https://www.jpmorgan.com/investment-banking/corporate-finance-advisory

[^2]: https://www.jpmorgan.com/investment-banking/debt-capital-markets

[^3]: https://www.jpmorgan.com/about-us/corporate-news/2025/jpmorgan-increases-direct-lending-commitment-to-50-billion
