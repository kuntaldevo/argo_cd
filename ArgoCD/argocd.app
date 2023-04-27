apiVersion: argoproj.io/v1alpha1
metadata:
  name: Default-App
spec:
  destination:
    namespace: argocd
    server: 'https://kubernetes.default.svc'
  source:
    path: application/customer/t-shirt/kubernetes
    repoURL: 'http://10.241.101.13/root/devops-application'
    targetRevision: HEAD
    directory:
      recurse: true
  project: default
  syncPolicy:
    automated:
      prune: false
      selfHeal: true