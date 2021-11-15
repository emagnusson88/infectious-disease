library(shiny)
library(coronavirus)
library(ggplot2)
library(dplyr)
library(readr)
library(reticulate)

# Get daily updates from GitHub
#devtools::install_github("RamiKrispin/coronavirus",force=TRUE)
#update_dataset()

covid <- coronavirus

# See above for the definitions of ui and server
fluidPage(
  titlePanel("Covid Forecasting"),
  
  sidebarLayout(
    sidebarPanel(
      selectizeInput("countryInput","Country",
                     choices=unique(covid$country),
                     selected="US", multiple=FALSE),
      radioButtons("dataSource","Data Source:", c("Upload Data","Use Pre-loaded Data")),
      uiOutput("ui"),
      actionButton("update","Update Chart")
    ),
    
    mainPanel(
      plotOutput("covidplot"), 
      h3('Python script output'),
      verbatimTextOutput('message')
    )
  )
)