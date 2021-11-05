pipeline {
  agent none
  stages {
    stage('Unit Tests') {
      parallel {
        stage('3.6'){
          agent{
            dockerfile {
              filename 'python36.Dockerfile'
              dir '.ci'
              args '-v $HOME/.cache/pip:/home/jenkins/.cache/pip'
            }
          }
          steps {
              sh 'pip install pip --upgrade --progress-bar off'
              sh 'pip install .[all] --progress-bar off'
              sh '~/.local/bin/pytest -v --junitxml=reports/result.xml --cov=prototorch/ --cov-report=xml:reports/coverage.xml'
              cobertura coberturaReportFile: 'reports/coverage.xml'
              junit 'reports/**/*.xml'
          }
        }
        stage('3.10'){
          agent{
            dockerfile {
              filename 'python310.Dockerfile'
              dir '.ci'
              args '-v $HOME/.cache/pip:/home/jenkins/.cache/pip'
            }
          }
          steps {
            sh 'pip install pip --upgrade --progress-bar off'
            sh 'pip install .[all] --progress-bar off'
            sh '~/.local/bin/pytest -v --junitxml=reports/result.xml --cov=prototorch/ --cov-report=xml:reports/coverage.xml'
            cobertura coberturaReportFile: 'reports/coverage.xml'
            junit 'reports/**/*.xml'
        }
        }
      }
    }
  }
}
