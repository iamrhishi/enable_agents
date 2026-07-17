import React from 'react';
import './NetworkProfileCard.css';

function normalizeSkills(skills) {
  if (!skills) return null;
  if (Array.isArray(skills)) return skills.join(', ');
  return String(skills);
}

export function normalizeSearchProfile(result, index = 0) {
  const name = result.name || result.Name || result.full_name || 'Unknown';
  const lastname = result.Surname || '';
  const full_name = `${name} ${lastname}`.trim();
  return {
    id: result.id || `profile_${index}_${full_name.replace(/\s+/g, '_')}`,
    full_name,
    company: result.company || result.Company || result.organization || 'Unknown',
    title: result['Job title'] || result.title || result.Title || result.position || result.role || 'Unknown',
    location: result.location || result.Location || result.city || result.City || '',
    skills: normalizeSkills(result.skills || result.Skills || result.required_skills || result.technologies),
    email: result.email,
    linkedin: result.linkedin,
    raw: result,
  };
}

function NetworkProfileCard({ profile, onSave, onMessage }) {
  const linkedinSearchUrl = `https://www.linkedin.com/search/results/people/?keywords=${encodeURIComponent(profile.full_name)}`;

  return (
    <article className="network-profile-card">
      <div className="network-profile-card-header">
        <strong className="network-profile-name">{profile.full_name}</strong>
        <div className="network-profile-actions">
          <a href={linkedinSearchUrl} target="_blank" rel="noopener noreferrer" title="Search on LinkedIn">
            <img src="/assets/icons/linkedin.png" alt="" className="network-profile-action-icon" />
          </a>
          <button type="button" className="network-profile-action-btn" onClick={() => onSave?.(profile)} title="Save to favorites">
            <img src="/assets/icons/save.png" alt="" className="network-profile-action-icon" />
          </button>
          <button type="button" className="network-profile-action-btn" onClick={() => onMessage?.(profile)} title="Generate outreach message">
            <img src="/assets/icons/message.png" alt="" className="network-profile-action-icon" />
          </button>
        </div>
      </div>
      <dl className="network-profile-details">
        <div><dt>Company</dt><dd>{profile.company}</dd></div>
        <div><dt>Title</dt><dd>{profile.title}</dd></div>
        {profile.location && <div><dt>Location</dt><dd>{profile.location}</dd></div>}
        {profile.skills && profile.skills.length < 80 && (
          <div><dt>Skills</dt><dd>{profile.skills}</dd></div>
        )}
      </dl>
    </article>
  );
}

export function NetworkSearchResults({ results = [], onSave, onMessage }) {
  const profiles = results.slice(0, 5).map((r, i) => normalizeSearchProfile(r, i));
  return (
    <div className="network-search-results">
      {profiles.map((profile) => (
        <NetworkProfileCard
          key={profile.id}
          profile={profile}
          onSave={onSave}
          onMessage={onMessage}
        />
      ))}
    </div>
  );
}

export default NetworkProfileCard;
