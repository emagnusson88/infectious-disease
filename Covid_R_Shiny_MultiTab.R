library(shinydashboard)
library(shiny)
library(coronavirus)
library(ggplot2)
library(dplyr)
library(readr)
library(reticulate)
library(dashboardthemes)


# Get daily updates from GitHub
covid <- read.csv('https://raw.githubusercontent.com/RamiKrispin/coronavirus/master/csv/coronavirus.csv')

# format date
covid$date <- as.Date(covid$date, format =  "%Y-%m-%d")

# define sidebar tabs 
sidebar <- dashboardSidebar(
  sidebarMenu(
    menuItem("About this App", tabName = "about", icon = icon("info-circle")),
    menuItem("Data load & display", tabName = "data", icon = icon("chart-line")),
    menuItem("Model & parameter selection", tabName = "model", icon = icon("tools")),
    menuItem("Model forecast & display", tabName = "forecast", icon = icon("chart-line"))
  )
)

# define tab body contents
body <- dashboardBody(
  
  shinyDashboardThemes(
    # other theme options: https://github.com/nik01010/dashboardthemes
    #theme = "grey_light"  
    theme = "onenote"
  ),
  
  
  
  tabItems(
    tabItem(tabName = "about",
            h4("Covid Forecasting"),
            p("This app has been designed to facilitate modeling and forecasting of infectious diseases including COVID..."),
            h4("New heading"),
            p("New paragraph")
    ),
    
    tabItem(tabName = "data", 
            fluidRow(
              box(radioButtons("dataSource","Data Source:", c("Upload Data","Use Pre-loaded Data")),
                  uiOutput("ui"),
                  actionButton("update","Update Chart")
                  ),
              
              box(selectizeInput("countryInput","Country:",
                                 choices=unique(covid$country),
                                 selected="US", multiple=FALSE)
                  )
            ),
            
            fluidRow(
              plotOutput("covidplot")
            )
    )
  )
)

# Combine elements of dashboardPage
ui <- dashboardPage(
  dashboardHeader(title = "Covid Forecasting"),
  sidebar,
  body
)


server <- function(input,output){
  output$ui <- renderUI({
    if (is.null(input$dataSource))
      return()
    
    switch(input$dataSource,
           "Upload Data" = fileInput("dataFile","Choose CSV File",accept = ".csv")
    )
  })
  
#  reticulate::source_python('python_test.py')
  
#  output$message <- renderText({
#    return(python_test())
#  })
  
  
  c <- eventReactive(input$update,{
    covid %>%
      filter(country == input$countryInput,
             type == "confirmed",
             date >= '2020-01-22',
             date <= '2021-10-31')
  })
  
  output$covidplot <- renderPlot({
    ggplot(c(), aes(x=date,y=cases,color=country)) +
      geom_point() +
      geom_line() +
      xlab("Date") +
      ylab("Cases") +
      ggtitle("Cases Over Time")
  })
  
}
  

shinyApp(ui = ui, server = server)