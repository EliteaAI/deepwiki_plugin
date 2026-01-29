{{/*
Expand the name of the chart.
*/}}
{{- define "deepwiki.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "deepwiki.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "deepwiki.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "deepwiki.labels" -}}
helm.sh/chart: {{ include "deepwiki.chart" . }}
{{ include "deepwiki.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "deepwiki.selectorLabels" -}}
app.kubernetes.io/name: {{ include "deepwiki.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "deepwiki.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "deepwiki.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the service location URL
*/}}
{{- define "deepwiki.serviceLocationUrl" -}}
{{- if .Values.config.serviceLocationUrl }}
{{- .Values.config.serviceLocationUrl }}
{{- else }}
{{- printf "http://%s.%s.svc.cluster.local:%d" (include "deepwiki.fullname" .) .Release.Namespace (int .Values.service.port) }}
{{- end }}
{{- end }}

{{/*
Return the PVC name
*/}}
{{- define "deepwiki.pvcName" -}}
{{- if .Values.storage.existingClaim }}
{{- .Values.storage.existingClaim }}
{{- else }}
{{- include "deepwiki.fullname" . }}-data
{{- end }}
{{- end }}

{{/*
Determine deployment mode
*/}}
{{- define "deepwiki.mode" -}}
{{- if and .Values.api.enabled .Values.worker.enabled }}
{{- "split" }}
{{- else if .Values.combined.enabled }}
{{- "combined" }}
{{- else }}
{{- "combined" }}
{{- end }}
{{- end }}
