const policies = {
  privacy: {
    title: "Privacy Notice",
    sections: [
      ["Information collected", "Curalink stores your account name and email, plus the disease, research intent, general location, and messages you choose to provide. Do not enter patient names, birth dates, exact addresses, medical-record numbers, contact details, or other identifiable patient information."],
      ["How information is used", "Research context is used to search medical-literature services and generate personalized research summaries. Queries and retrieved evidence may be processed by the configured HuggingFace or Cloudflare AI provider. Prompt and answer content is not exported to observability services."],
      ["Service providers", "Curalink uses Render for hosting, MongoDB Atlas for account and session storage, Upstash Redis for temporary caches, and PubMed, OpenAlex, ClinicalTrials.gov, and OpenStreetMap for research retrieval and general-location search. Those services process requests under their own policies."],
      ["Retention", "Research sessions and messages expire after 90 days. Query and generated-response caches expire within 24 hours; document embeddings expire within seven days. Account information remains until you delete the account."],
      ["Your controls", "You can delete individual research sessions or permanently delete your account from the sidebar. Account deletion removes active account, session, message, and user-scoped cache records."],
      ["Public-beta boundary", "Curalink is a research-information service, not a medical record system. It is not intended to receive protected or directly identifiable patient information."],
    ],
  },
  terms: {
    title: "Terms of Use",
    sections: [
      ["Research use only", "Curalink helps users discover and summarize medical research using de-identified context. It does not diagnose conditions, select treatments, prescribe medication, determine clinical-trial eligibility, or replace a qualified healthcare professional."],
      ["No emergency use", "Do not use Curalink for emergencies or urgent assessment. Contact local emergency services or go to the nearest emergency department."],
      ["Verify original sources", "AI-generated summaries and personalized recommendations can be incomplete or incorrect. Review the linked publications and consult a qualified healthcare professional before making a health decision."],
      ["Your responsibilities", "Do not submit identifiable patient information, attempt unauthorized access, misuse another person's account, or use the service to cause harm."],
      ["Public beta", "The service may change, experience delays, or be unavailable. Research results are provided without a guarantee of completeness, accuracy, or fitness for clinical decision-making."],
      ["Ending use", "You may stop using Curalink at any time and permanently delete your account and active data from the sidebar."],
    ],
  },
};

export default function LegalPage({ type, onBack }) {
  const policy = policies[type] || policies.privacy;
  return (
    <main className="legal-page">
      <article className="legal-card">
        <button className="legal-back" onClick={onBack}>&larr; Back</button>
        <p className="legal-kicker">CURALINK PUBLIC BETA</p>
        <h1>{policy.title}</h1>
        <p className="legal-updated">Last updated: September 25, 2026</p>
        {policy.sections.map(([heading, body]) => (
          <section key={heading}>
            <h2>{heading}</h2>
            <p>{body}</p>
          </section>
        ))}
        <p className="legal-contact">
          Questions can be submitted through the project&apos;s{" "}
          <a href="https://github.com/VIVPM/curalink-medical-assistant/issues" target="_blank" rel="noreferrer">
            support channel
          </a>.
        </p>
      </article>
    </main>
  );
}
