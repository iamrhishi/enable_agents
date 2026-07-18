/**
 * Demo Data Loader
 *
 * Centralized loader for all agent demo data.
 * JSON files can be easily edited to change demo content.
 *
 * Usage:
 *   import { getDemoData, getDemoProjects, getAllDemoData } from '../data/demo';
 *
 *   // Get data for a specific agent
 *   const salesData = getDemoData('salesHelper');
 *
 *   // Get demo projects
 *   const projects = getDemoProjects();
 *
 *   // Get all demo data
 *   const allData = getAllDemoData();
 */

import marketResearchData from './marketResearch.json';
import salesHelperData from './salesHelper.json';
import contentMarketingData from './contentMarketing.json';
import communityNetworkData from './communityNetwork.json';
import eventNetworkingData from './eventNetworking.json';
import executiveAssistantData from './executiveAssistant.json';
import dataInsightsData from './dataInsights.json';
import projectsData from './projects.json';

/**
 * All demo data organized by agent key
 */
export const DEMO_DATA = {
  marketResearch: marketResearchData,
  salesHelper: salesHelperData,
  contentMarketing: contentMarketingData,
  communityNetwork: communityNetworkData,
  eventNetworking: eventNetworkingData,
  executiveAssistant: executiveAssistantData,
  dataInsights: dataInsightsData,
};

/**
 * Get demo data for a specific agent
 * @param {string} agentKey - Agent identifier (e.g., 'salesHelper')
 * @returns {object} Demo data for the agent, or empty object if not found
 */
export function getDemoData(agentKey) {
  return DEMO_DATA[agentKey] || {};
}

/**
 * Get demo projects
 * @returns {array} Array of demo projects
 */
export function getDemoProjects() {
  return projectsData.projects || [];
}

/**
 * Get all demo data
 * @returns {object} All demo data organized by agent key
 */
export function getAllDemoData() {
  return DEMO_DATA;
}

/**
 * Get demo projects with agent data populated
 * @returns {array} Projects with data field populated from agent demo data
 */
export function getDemoProjectsWithData() {
  return projectsData.projects.map(project => ({
    ...project,
    data: project.agents.reduce((acc, agentKey) => {
      const agentData = DEMO_DATA[agentKey];
      if (agentData) {
        acc[agentKey] = agentData;
      }
      return acc;
    }, {}),
  }));
}

/**
 * Agent display names (for UI)
 */
export const AGENT_NAMES = {
  marketResearch: 'Market Research',
  salesHelper: 'Sales Helper',
  contentMarketing: 'Content Marketing',
  communityNetwork: 'Community Network',
  eventNetworking: 'Event Networking',
  executiveAssistant: 'Executive Assistant',
  dataInsights: 'Data Insights',
};

export default {
  DEMO_DATA,
  getDemoData,
  getDemoProjects,
  getAllDemoData,
  getDemoProjectsWithData,
  AGENT_NAMES,
};
