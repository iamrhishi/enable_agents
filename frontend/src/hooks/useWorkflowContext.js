/**
 * useWorkflowContext Hook
 *
 * Provides workflow context to agents when opened from a workflow.
 * Agents use this to:
 * 1. Check if they're running in workflow context
 * 2. Save their outputs back to the workflow stage
 * 3. Access workflow/stage metadata
 *
 * Usage:
 *   const { isInWorkflow, stageId, saveStageData, workflowName } = useWorkflowContext();
 *
 *   // When agent completes work:
 *   await saveStageData({
 *     suppliers_found: 12,
 *     market_analysis: 'Complete',
 *     ...
 *   });
 */
import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { API_CONFIG } from '../config/apiConfig';
import { authJsonHeaders } from '../core/authHeaders';
import { showToast } from '../core/toast';

// Mapping from NEW stage IDs to OLD stage IDs for backward compatibility
// (workflows created before stage rename still have old keys in stageStates)
const STAGE_ID_MIGRATION = {
  supplier_discovery: 'requirement_capture',
  document_analysis: 'market_research_analysis',
  rfq_outreach: 'supplier_outreach',
  response_analysis: 'response_tracking',
  // qualification_audit unchanged
  selection_tasks: 'final_selection',
};

export function useWorkflowContext() {
  const [searchParams] = useSearchParams();
  const workflowId = searchParams.get('workflow');
  const stageId = searchParams.get('stage');
  const viewMode = searchParams.get('view'); // 'history' or 'run'

  const [workflowData, setWorkflowData] = useState(null);
  const [stageData, setStageData] = useState(null);
  const [stageStatus, setStageStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const isInWorkflow = Boolean(workflowId && stageId);
  // Auto-detect history view: explicit URL param OR stage is completed
  const isHistoryView = viewMode === 'history' || stageStatus === 'completed';
  const isRunView = viewMode === 'run' || (!viewMode && isInWorkflow && !isHistoryView);

  // Fetch workflow info when in workflow context
  useEffect(() => {
    if (!isInWorkflow) return;

    setLoading(true);
    fetch(`${API_CONFIG.BASE_URL}/api/workflows/instances/${workflowId}`, {
      headers: authJsonHeaders(),
    })
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          setWorkflowData({
            id: data.instance.id,
            name: data.instance.name,
            templateName: data.instance.templateName,
            status: data.instance.status,
            context: data.instance.context || {},
            projectId: data.instance.projectId,
          });
          // Get current stage data if any
          // Try new stage ID first, then fall back to old ID for migrated workflows
          let currentStageState = data.instance.stageStates?.[stageId];
          const usedOldId = !currentStageState && STAGE_ID_MIGRATION[stageId];
          if (usedOldId) {
            currentStageState = data.instance.stageStates?.[STAGE_ID_MIGRATION[stageId]];
          }

          // Debug logging for troubleshooting
          console.log('[useWorkflowContext] Stage lookup:', {
            stageId,
            usedOldId: usedOldId ? STAGE_ID_MIGRATION[stageId] : null,
            foundState: Boolean(currentStageState),
            stageStatus: currentStageState?.status,
            stageData: currentStageState?.data,
            allStageStates: Object.keys(data.instance.stageStates || {}),
          });

          if (currentStageState) {
            setStageData(currentStageState.data || {});
            setStageStatus(currentStageState.status || null);
          }
        }
      })
      .catch(err => console.error('Error loading workflow context:', err))
      .finally(() => setLoading(false));
  }, [workflowId, stageId, isInWorkflow]);

  /**
   * Save data to the current workflow stage.
   * This persists agent outputs back to the workflow.
   *
   * @param {Object} data - Key-value pairs to save
   * @param {Object} options - { merge: true } to merge with existing data (default), false to replace
   * @returns {Promise<boolean>} - Success status
   */
  const saveStageData = useCallback(async (data, options = { merge: true }) => {
    if (!isInWorkflow) {
      console.warn('saveStageData called but not in workflow context');
      return false;
    }

    if (!data || Object.keys(data).length === 0) {
      console.warn('saveStageData called with empty data');
      return false;
    }

    setSaving(true);
    try {
      const method = options.merge ? 'PATCH' : 'POST';
      const res = await fetch(
        `${API_CONFIG.BASE_URL}/api/workflows/instances/${workflowId}/stages/${stageId}/data`,
        {
          method,
          headers: authJsonHeaders(),
          body: JSON.stringify({ data }),
        }
      );
      const result = await res.json();

      if (result.success) {
        // Update local state
        setStageData(prev => options.merge ? { ...prev, ...data } : data);
        showToast('Progress saved to workflow', 'success');
        return true;
      } else {
        console.error('Failed to save stage data:', result.error);
        showToast(result.error || 'Failed to save progress', 'error');
        return false;
      }
    } catch (err) {
      console.error('Error saving stage data:', err);
      showToast('Error saving progress', 'error');
      return false;
    } finally {
      setSaving(false);
    }
  }, [workflowId, stageId, isInWorkflow]);

  /**
   * Get a value from workflow context (data from previous stages)
   */
  const getContextValue = useCallback((key) => {
    return workflowData?.context?.[key];
  }, [workflowData]);

  /**
   * Get all context data from workflow
   */
  const getContext = useCallback(() => {
    return workflowData?.context || {};
  }, [workflowData]);

  return {
    // State
    isInWorkflow,
    isHistoryView,
    isRunView,
    loading,
    saving,

    // Workflow info
    workflowId,
    stageId,
    workflowName: workflowData?.name,
    templateName: workflowData?.templateName,
    workflowStatus: workflowData?.status,
    projectId: workflowData?.projectId,

    // Data
    stageData,
    context: workflowData?.context || {},

    // Actions
    saveStageData,
    getContextValue,
    getContext,
  };
}

export default useWorkflowContext;
