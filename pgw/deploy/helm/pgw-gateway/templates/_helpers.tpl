{{/* Chart name. */}}
{{- define "pgw-gateway.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Fully qualified app name. */}}
{{- define "pgw-gateway.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/* Selector labels. */}}
{{- define "pgw-gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pgw-gateway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* Common labels. */}}
{{- define "pgw-gateway.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "pgw-gateway.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: gateway
app.kubernetes.io/part-of: pgw
{{- end -}}

{{/* Service account name. */}}
{{- define "pgw-gateway.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "pgw-gateway.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* Container image reference. */}}
{{- define "pgw-gateway.image" -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}

{{/* The Secret holding admin token, KEK and service token (required). */}}
{{- define "pgw-gateway.secretName" -}}
{{- required "secrets.existingSecret is required: create a Secret with the admin token, KEK and compute-service token" .Values.secrets.existingSecret -}}
{{- end -}}

{{/* Name of the state PVC. */}}
{{- define "pgw-gateway.stateClaim" -}}
{{- if .Values.state.persistence.existingClaim -}}
{{- .Values.state.persistence.existingClaim -}}
{{- else -}}
{{- printf "%s-state" (include "pgw-gateway.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "pgw-gateway.counterClaim" -}}
{{- if .Values.counterVolume.existingClaim -}}
{{- .Values.counterVolume.existingClaim -}}
{{- else -}}
{{- printf "%s-counter" (include "pgw-gateway.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/* URL scheme of the gateway listener. */}}
{{- define "pgw-gateway.scheme" -}}
{{- if .Values.tls.enabled -}}HTTPS{{- else -}}HTTP{{- end -}}
{{- end -}}
