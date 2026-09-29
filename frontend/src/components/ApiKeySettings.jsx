// Tab-scoped provider credentials dialog for privately funded research queries.

import { useEffect, useRef, useState } from "react";
import { authFetch } from "../session";

const API = `${import.meta.env.VITE_API_URL || ""}/api`;

export default function ApiKeySettings({ current, onSave, onRemove, onClose }) {
  const dialog = useRef(null);
  const [provider, setProvider] = useState(null);
  const [form, setForm] = useState({
    hfToken: current?.hfToken || "",
    cfAccountId: current?.cfAccountId || "",
    cfToken: current?.cfToken || "",
  });
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const node = dialog.current;
    if (!node.open) node.showModal();
    let active = true;
    authFetch(`${API}/account/keys/provider`)
      .then((response) => { if (!response.ok) throw new Error("Provider unavailable"); return response.json(); })
      .then((data) => { if (active) setProvider(data.provider); })
      .catch(() => { if (active) setError("Could not load provider requirements. Reopen Settings to retry."); });
    return () => { active = false; };
  }, []);

  const save = async (event) => {
    event.preventDefault();
    setError("");
    setSaving(true);
    const keys = provider === "cloudflare" ? form : { hfToken: form.hfToken, cfAccountId: "", cfToken: "" };
    try {
      const response = await authFetch(`${API}/account/keys/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(keys),
      });
      if (!response.ok) {
        const data = await response.json();
        setError(data.error || "Could not validate these API keys.");
        return;
      }
      if (onSave(keys)) nodeClose();
      else setError("Browser session storage is unavailable; keys were not saved.");
    } catch {
      setError("Credential verification is unavailable. Try again later.");
    } finally {
      setSaving(false);
    }
  };

  const nodeClose = () => dialog.current?.close();

  return (
    <dialog ref={dialog} className="api-keys-dialog" onClose={onClose} aria-labelledby="api-keys-title">
      <form onSubmit={save}>
        <div className="api-keys-heading">
          <h2 id="api-keys-title">Your API keys</h2>
          <button type="button" onClick={nodeClose} aria-label="Close settings">×</button>
        </div>
        <p>Use your own provider keys for research queries. Your daily question credits will be hidden; safety and request-rate limits still apply.</p>
        <p>Keys stay in this browser tab and are sent only to the API for validation and inference. Closing the tab removes them.</p>
        <label htmlFor="hf-key">Hugging Face token <span>(required for research ranking{provider === "huggingface" ? " and answers" : ""})</span></label>
        <input id="hf-key" type={showKey ? "text" : "password"} autoComplete="off" value={form.hfToken}
          onChange={(event) => setForm({ ...form, hfToken: event.target.value })} required maxLength={1024} />
        {provider === "cloudflare" && (
          <>
            <label htmlFor="cf-account">Cloudflare account ID</label>
            <input id="cf-account" type="text" autoComplete="off" value={form.cfAccountId}
              onChange={(event) => setForm({ ...form, cfAccountId: event.target.value })} required maxLength={64} />
            <label htmlFor="cf-key">Cloudflare Workers AI token</label>
            <input id="cf-key" type={showKey ? "text" : "password"} autoComplete="off" value={form.cfToken}
              onChange={(event) => setForm({ ...form, cfToken: event.target.value })} required maxLength={1024} />
          </>
        )}
        <label className="api-keys-reveal"><input type="checkbox" checked={showKey}
          onChange={(event) => setShowKey(event.target.checked)} /> Show keys</label>
        {error && <p role="alert" className="api-keys-error">{error}</p>}
        <div className="api-keys-actions">
          {current && <button type="button" className="api-keys-remove" onClick={() => { onRemove(); nodeClose(); }}>Remove keys</button>}
          <button type="button" className="api-keys-cancel" onClick={nodeClose}>Cancel</button>
          <button type="submit" className="api-keys-save" disabled={saving || !provider}>
            {saving ? "Checking…" : "Validate and use keys"}
          </button>
        </div>
      </form>
    </dialog>
  );
}
